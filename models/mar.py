import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import logging
import copy
import ocnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.condition import ConditionCrossAttn
from models.octformer import OctFormer, OctreeT
from models.positional_embedding import SinPosEmb
from models.vae import DiagonalGaussian
from utils.utils import seq2octree, sample, depth2batch, batch2depth, \
    get_depth2batch_indices, get_batch_id, export_octree
from utils.distributed import get_rank
logger = logging.getLogger(__name__)


class MAR(nn.Module):
  """ Masked Autoencoder with VisionTransformer backbone
  """

  def __init__(self,
               num_embed=256,
               num_heads=8,
               num_blocks=8,
               num_classes=1,
               patch_size=4096,
               dilation=2,
               buffer_size=64,
               drop_rate=0.1,
               pos_emb_type="SinPosEmb",
               use_checkpoint=True,
               use_swin=True,
               num_vq_blocks=2,
               mask_ratio_min=0.7,
               start_temperature=1.0,
               remask_stage=0.9,
               vqvae_config=None,
               condition_type="None",
               condition_policy="concat",
               num_iters=256,
               context_dim=512,
               **kwargs):
    super(MAR, self).__init__()
    self.vqvae_config = vqvae_config
    self.num_embed = num_embed
    self.num_heads = num_heads
    self.num_blocks = num_blocks
    self.patch_size = patch_size
    self.dilation = dilation
    self.buffer_size = buffer_size if condition_type in ['None', 'category'] else 0
    self.drop_rate = drop_rate
    self.pos_emb_type = pos_emb_type
    self.use_checkpoint = use_checkpoint
    self.use_swin = use_swin
    self.use_flex = False
    self.num_vq_blocks = num_vq_blocks
    self.start_temperature = start_temperature
    self.remask_stage = remask_stage
    self.num_iters = num_iters
    self.condition_type = condition_type
    self.split_size = 2  # 0/1 indicates the split signals
    self.num_vq_embed = vqvae_config.embedding_channels
    self.vq_size = vqvae_config.embedding_sizes
    self.vq_groups = vqvae_config.quantizer_group

    self.split_emb = nn.Embedding(self.split_size, num_embed)
    self.class_emb = nn.Embedding(num_classes, num_embed)
    
    self.condition_policy = condition_policy
    if condition_type not in ['None', 'category']:
      self.mask_token = nn.Parameter(torch.zeros(1, num_embed))
      if condition_policy == 'cross_attn':
        self.cond_encoder = ConditionCrossAttn(
          num_embed=num_embed, num_heads=num_heads, dropout=drop_rate, context_dim=context_dim)
      else: # Useless, just for reading existing checkpoint that are saved before cleaning the code
        self.cond_preln = nn.LayerNorm(num_embed)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=num_embed, 
            num_heads=num_heads,
            dropout=drop_rate,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        self.cond_postln = nn.LayerNorm(num_embed)
        
    self.vq_proj = nn.Linear(self.num_vq_embed, num_embed)

    self._init_blocks()

    self.split_head = nn.Linear(num_embed, self.split_size)
    self.vq_head = nn.Linear(num_embed, self.vq_size * self.vq_groups)

    self.mask_ratio_generator = stats.truncnorm(
        (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

    self.apply(self._init_weights)  # initialize weights
    if condition_type not in ['None', 'category']:
      self._init_weights(self.mask_token)
    logger.info("number of parameters: %e", sum(p.numel()
                for p in self.parameters()))

  def _init_blocks(self):
    self.blocks = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin,
        cond_interval=3)
    self.ln_x = nn.LayerNorm(self.num_embed)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Parameter):
      module.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

  def get_mask(self, seq_len, orders, mask_rate=None):
    # generate token mask
    if mask_rate is None:
      mask_rate = self.mask_ratio_generator.rvs(1)[0]
    # mask_rate = 1.0
    num_masked_tokens = max(int(np.ceil(seq_len * mask_rate)), 1)
    return self.mask_by_order(num_masked_tokens, orders)

  def mask_by_order(self, mask_len, orders):
    mask = torch.zeros(orders.shape[0], device=orders.device).long()
    mask[orders[:mask_len]] = 1
    return mask

  def random_masking(self, x, mask, octree, depth_list):
    if self.condition_type in ['None', 'category']:
      batch_id = get_batch_id(octree, depth_list)
      mask_tokens = self.cond[batch_id]
    elif self.condition_type == 'image':
      # mask_tokens = self.mask_token.repeat(x.shape[0], 1)
      mask_tokens = self.cond.reshape(-1, self.num_embed).mean(dim=0, keepdim=True)
      mask_tokens = mask_tokens.repeat(x.shape[0], 1)
    x = torch.where(mask.bool().unsqueeze(1), mask_tokens, x)
    return x

  def forward_blocks(self, x, octree: OctreeT, blocks, use_cond=False):
    x = depth2batch(x, octree.indices)
    if self.condition_policy == 'cross_attn' and use_cond:
      x = blocks(x, octree, cond=self.cond, cond_enc=self.cond_encoder)
    else:
      x = blocks(x, octree)
    x = batch2depth(x, octree.indices)
    return x

  def forward_model(self, x, octree, depth_low, depth_high, mask, nnum_split):
    depth_list = list(range(depth_low, depth_high + 1))

    buffer = self.cond.reshape(octree.batch_size, 1, -1)
    buffer = buffer.repeat(1, self.buffer_size, 1).reshape(-1, self.num_embed)
    mask_buffer = torch.zeros(buffer.shape[0], device=x.device).bool()
    x = torch.cat([buffer, x], dim=0)
    mask = torch.cat([mask_buffer, mask], dim=0)

    octreeT = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list, use_swin=self.use_swin, buffer_size=self.buffer_size)
    x = self.forward_blocks(x, octreeT, self.blocks)
    x = self.ln_x(x)
    return x

  def forward(self, octree_in, depth_low, depth_high, condition=None, split=None, vqvae=None):
    batch_size = octree_in.batch_size

    if self.condition_type == "None":
      condition = torch.zeros(batch_size).long().to(octree_in.device)
      self.cond = self.class_emb(condition)  # 1 x C
    elif self.condition_type == "category":
      self.cond = self.class_emb(condition)
    elif self.condition_type == 'image':
      self.cond = condition

    split_token_embeddings = self.split_emb(split)  # (nnum_split, C)
    targets_split = split.clone().detach()
    nnum_split = split_token_embeddings.shape[0]

    with torch.no_grad():
      vq_code = vqvae.extract_code(octree_in)
      zq, indices, _ = vqvae.quantizer(vq_code)
      targets_vq = copy.deepcopy(indices)
    vq_token_embeddings = self.vq_proj(zq)
    nnum_vq = vq_token_embeddings.shape[0]

    # x_token_embeddings: (nnum, C)
    x_token_embeddings = torch.cat(
        [split_token_embeddings, vq_token_embeddings], dim=0)

    seq_len = x_token_embeddings.shape[0]
    orders = torch.randperm(seq_len, device=x_token_embeddings.device)
    mask = self.get_mask(seq_len, orders).bool()
    x_token_embeddings = self.random_masking(
        x_token_embeddings, mask, octree_in, list(range(depth_low, depth_high + 1)))

    # forward model
    x = self.forward_model(
        x_token_embeddings, octree_in, depth_low, depth_high, mask, nnum_split)

    output = {}
    # split accuracy
    mask_split = mask[:nnum_split]
    split_logits = self.split_head(x[:nnum_split])
    output['split_loss'] = F.cross_entropy(
        split_logits[mask_split], targets_split[mask_split])
    with torch.no_grad():
      correct_top1 = self.get_correct_topk(
          split_logits[mask_split], targets_split[mask_split], topk=1)
      output['split_accuracy'] = correct_top1.sum().float() / \
          mask_split.sum().float()

    # VQ accuracy
    mask_vq = mask[-nnum_vq:]
    vq_logits = self.vq_head(x[-nnum_vq:])
    vq_logits = vq_logits[mask_vq].reshape(-1, self.vq_size)
    targets_vq = targets_vq[mask_vq].reshape(-1)
    output['vq_loss'] = F.cross_entropy(
        vq_logits, targets_vq)
    # Top-k Accuracy
    with torch.no_grad():
      correct_top5 = self.get_correct_topk(vq_logits, targets_vq, topk=5)
      output['top5_accuracy'] = correct_top5.sum().float() / \
          (mask_vq.sum().float() * self.vq_groups)
    return output

  def get_correct_topk(self, logits, targets, topk=1):
    topk = torch.topk(logits, topk, dim=-1).indices
    correct_topk = topk.eq(targets.unsqueeze(-1).expand_as(topk))
    return correct_topk

  def get_remask(self, logits, tokens, mask, remask_prob=0.2, topk=1):
    remask = torch.rand_like(mask.float()) < remask_prob
    correct_topk = self.get_correct_topk(logits, tokens, topk=topk)
    correct_topk = correct_topk.any(dim=-1)
    if len(correct_topk.shape) > 1:
      correct_topk = correct_topk.all(dim=-1)
    remask = remask & ~correct_topk & ~mask
    return remask

  @torch.no_grad()
  def generate(self, octree, depth_low, depth_high, token_embeddings=None, condition=None, vqvae=None):
    batch_size = octree.batch_size
    if self.condition_type == "None":
      condition = torch.zeros(batch_size).long().to(octree.device)
      self.cond = self.class_emb(condition)  # 1 x C
    elif self.condition_type == "category":
      self.cond = self.class_emb(condition)
    elif self.condition_type == 'image':
      self.cond = condition

    if token_embeddings is None:
      token_embeddings = torch.empty((0, self.num_embed), device=octree.device)
    vq_code = torch.empty((0, self.num_vq_embed), device=octree.device)

    for d in range(depth_low, depth_high + 1):
      nnum_d = octree.nnum[d]
      nnum_split = sum([octree.nnum[i]
                       for i in range(depth_low, min(d + 1, depth_high))])

      mask = torch.zeros(token_embeddings.shape[0], device=octree.device).bool()
      mask_d = torch.ones(nnum_d, device=octree.device).bool()
      split_d = -1 * torch.ones(nnum_d, device=octree.device).long()
      vq_indices_d = -1 * \
          torch.ones((nnum_d, self.vq_groups), device=octree.device).long()
      vq_code_d = torch.zeros(nnum_d, self.num_vq_embed, device=octree.device)

      # fullly masked
      token_embedding_d = torch.zeros(
          nnum_d, self.num_embed, device=octree.device)
      token_embedding_d = self.random_masking(
          token_embedding_d, mask_d, octree, [d])
      orders = torch.randperm(nnum_d, device=octree.device)

      # set generate parameters
      num_iters = self.num_iters[d - depth_low] \
          if isinstance(self.num_iters, list) else self.num_iters
      start_temperature = self.start_temperature[d - depth_low] \
          if isinstance(self.start_temperature, list) else self.start_temperature

      for i in tqdm(range(num_iters)):
        x = torch.cat([token_embeddings, token_embedding_d], dim=0)
        x = self.forward_model(x, octree, depth_low, d, nnum_split=nnum_split,
                               mask=torch.cat([mask, mask_d]))
        x = x[-nnum_d:, :]

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_iters)
        mask_len = torch.Tensor([np.floor(nnum_d * mask_ratio)]).cuda()

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(
            torch.sum(mask_d, dim=-1, keepdims=True) - 1, mask_len)).long()

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next_d = self.mask_by_order(mask_len, orders).bool()

        if i >= num_iters - 1:
          mask_to_pred = mask_d.bool()
        else:
          mask_to_pred = torch.logical_xor(
              mask_d.bool(), mask_next_d.bool())
        mask_d = mask_next_d

        temperature = start_temperature * \
            ((num_iters - i) / num_iters)

        token_embedding_d[mask_to_pred] = 0.0
        if d < depth_high:
          split_logits = self.split_head(x)
          # remask tokens that have poor confidence
          if i > num_iters * self.remask_stage:
            remask = self.get_remask(split_logits, split_d, mask_d)
            mask_to_pred = mask_to_pred | remask
          ix = sample(split_logits[mask_to_pred], temperature=temperature)
          split_d[mask_to_pred] = ix
          token_embedding_d[mask_to_pred] += self.split_emb(ix)
        else:
          vq_logits = self.vq_head(x)
          if i > num_iters * self.remask_stage:
            vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)
            remask = self.get_remask(vq_logits, vq_indices_d, mask_d, topk=5)
            mask_to_pred = mask_to_pred | remask
          vq_logits = vq_logits[mask_to_pred].reshape(-1, self.vq_size)
          ix = sample(vq_logits, top_p=0.75, temperature=temperature)
          ix = ix.reshape(-1, self.vq_groups)
          vq_indices_d[mask_to_pred] = ix.long()
          with torch.no_grad():
            zq = vqvae.quantizer.extract_code(ix)
            vq_code_d[mask_to_pred] = zq.float()
            token_embedding_d[mask_to_pred] += self.vq_proj(zq)

      token_embeddings = torch.cat(
          [token_embeddings, token_embedding_d], dim=0)
      if d < depth_high:
        split_d = split_d.long()
        octree = seq2octree(octree, split_d, d, d + 1)
        # export_octree(
        # octree, d + 1, f"mytools/octree/depth{d+1}/", index=get_rank())
      else:
        vq_code = torch.cat([vq_code, vq_code_d], dim=0)

    return octree, vq_code


class MARUNet(MAR):
  def _init_blocks(self):
    self.blocks = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.ln_x = nn.LayerNorm(self.num_embed)

    self.vq_encoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_vq_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.downsample = ocnn.modules.OctreeConvGnRelu(
        self.num_embed, self.num_embed, group=32, kernel_size=[2], stride=2)

    self.vq_decoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_vq_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.upsample = ocnn.modules.OctreeDeconvGnRelu(
        self.num_embed, self.num_embed, group=32, kernel_size=[2], stride=2)

  def forward_model(self, x, octree, depth_low, depth_high, mask, nnum_split):
    # if only split signals, not use vq sample
    apply_vq_sample = nnum_split < x.shape[0]
    if apply_vq_sample:
      depth_list_main = list(range(depth_low, depth_high)) + [depth_high - 1]
    else:
      depth_list_main = list(range(depth_low, depth_high + 1))

    depth_list_vq = list(range(depth_low, depth_high + 1))
    octreeT_vq = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list_vq, use_swin=self.use_swin, use_flex=self.use_flex)
    x = self.forward_blocks(x, octreeT_vq, self.vq_encoder)
    if apply_vq_sample:
      x_split = x[:nnum_split]
      x_vq = x[nnum_split:]
      x_vq = self.downsample(x_vq, octree, depth_high)
      x = torch.cat([x_split, x_vq], dim=0)

    octreeT = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation,
        nempty=False, depth_list=depth_list_main, use_swin=self.use_swin)
    x = self.forward_blocks(
        x, octreeT, self.blocks)

    if apply_vq_sample:
      # skip connection
      x_split = x[:nnum_split]
      x_vq = x[nnum_split:] + x_vq
      x_vq = self.upsample(x_vq, octree, depth_high - 1)
      x = torch.cat([x_split, x_vq], dim=0)
    x = self.forward_blocks(x, octreeT_vq, self.vq_decoder)

    x = self.ln_x(x)
    return x


class MAREncoderDecoder(MAR):
  def _init_blocks(self):
    self.encoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin,
        cond_interval=3)
    self.encoder_ln = nn.LayerNorm(self.num_embed)

    self.decoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.decoder_ln = nn.LayerNorm(self.num_embed)

  def forward_model(self, x, octree, depth_low, depth_high, mask, nnum_split):
    batch_size = octree.batch_size
    depth_list = list(range(depth_low, depth_high + 1))

    # Add condition to the input
    if self.condition_type in ['None', 'category']:
      buffer = self.cond.reshape(octree.batch_size, 1, -1)
      buffer = buffer.repeat(1, self.buffer_size, 1).reshape(-1, self.num_embed)
      mask_buffer = torch.zeros(buffer.shape[0], device=x.device).bool()
      x = torch.cat([buffer, x], dim=0)
      mask = torch.cat([mask_buffer, mask], dim=0)
    elif self.condition_type == 'image' and self.condition_policy == 'concat':
      buffer = self.cond.squeeze(0)
      self.buffer_size = buffer.shape[0]
      x = torch.cat([buffer, x], dim=0)
      mask = torch.cat([torch.zeros(buffer.shape[0], device=x.device).bool(), mask], dim=0)
    
    x_enc = x.clone()
    x_enc = x_enc[~mask]
    
    if len(x_enc):
      octreeT_encoder = OctreeT(
          octree, x_enc.shape[0], self.patch_size, self.dilation, nempty=False,
          depth_list=depth_list, use_swin=self.use_swin, use_flex=self.use_flex,
          data_mask=mask, buffer_size=self.buffer_size)
      x_enc = self.forward_blocks(x_enc, octreeT_encoder, self.encoder, True)
      x_enc = self.encoder_ln(x_enc)
    x[~mask] = x_enc
    
    octreeT_decoder = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list, use_swin=self.use_swin, use_flex=self.use_flex,
        buffer_size=self.buffer_size)
    x = self.forward_blocks(x, octreeT_decoder, self.decoder)
    x = self.decoder_ln(x)
    x = x[batch_size * self.buffer_size:]

    return x


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
  """
  inputs: (N, 2) logits for negative and positive classes
  targets: (N) ground truth labels, with values in {0, 1}
  """
  BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
  # prevents nan as pt -> +inf when the initial loss is high
  pt = torch.exp(-BCE_loss)
  alpha_t = alpha * (1.0 - targets) + (1.0 - alpha) * targets
  F_loss = alpha_t * (1 - pt) ** gamma * BCE_loss
  return torch.mean(F_loss)
