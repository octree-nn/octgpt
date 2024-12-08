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
from models.octformer import OctFormer
from models.positional_embedding import SinPosEmb, OctreeConvPosEmb
from models.vae import DiagonalGaussian
from utils.utils import seq2octree, sample, depth2batch, batch2depth, get_depth2batch_indices, get_batch_id, export_octree
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
               drop_rate=0.1,
               pos_emb_type="SinPosEmb",
               use_checkpoint=True,
               use_swin=True,
               use_vq_sample=False,
               num_vq_blocks=2,
               mask_ratio_min=0.7,
               start_temperature=1.0,
               remask_stage=0.9,
               vqvae_config=None,
               num_iters=256,
               **kwargs):
    super(MAR, self).__init__()
    self.vqvae_config = vqvae_config
    self.num_embed = num_embed
    self.num_vq_embed = vqvae_config.embedding_channels
    self.num_blocks = num_blocks
    self.start_temperature = start_temperature
    self.remask_stage = remask_stage
    self.num_iters = num_iters
    self.split_size = 2  # 0/1 indicates the split signals
    self.vq_size = vqvae_config.embedding_sizes
    self.vq_groups = vqvae_config.quantizer_group
    self.use_vq_sample = use_vq_sample
    self.num_vq_blocks = num_vq_blocks

    self.split_emb = nn.Embedding(self.split_size, num_embed)
    self.class_emb = nn.Embedding(num_classes, num_embed)
    self.vq_proj = nn.Linear(self.num_vq_embed, num_embed)

    self.drop = nn.Dropout(drop_rate)
    self.blocks = OctFormer(
        channels=num_embed, num_blocks=num_blocks, num_heads=num_heads,
        patch_size=patch_size, dilation=dilation, attn_drop=drop_rate,
        proj_drop=drop_rate, pos_emb=eval(pos_emb_type), nempty=False,
        use_checkpoint=use_checkpoint, use_swin=use_swin)

    self.ln_x = nn.LayerNorm(num_embed)
    self.split_head = nn.Linear(num_embed, self.split_size)
    self.vq_head = nn.Linear(num_embed, self.vq_size * self.vq_groups)

    if self.use_vq_sample:
      self.vq_encoder = OctFormer(
          channels=num_embed, num_blocks=num_vq_blocks // 2, num_heads=num_heads,
          patch_size=patch_size, dilation=dilation, attn_drop=drop_rate,
          proj_drop=drop_rate, pos_emb=eval(pos_emb_type), nempty=False,
          use_checkpoint=use_checkpoint, use_swin=use_swin)
      self.downsample = ocnn.modules.OctreeConvGnRelu(
          num_embed, num_embed, group=32, kernel_size=[2], stride=2)

      self.vq_decoder = OctFormer(
          channels=num_embed, num_blocks=num_vq_blocks // 2, num_heads=num_heads,
          patch_size=patch_size, dilation=dilation, attn_drop=drop_rate,
          proj_drop=drop_rate, pos_emb=eval(pos_emb_type), nempty=False,
          use_checkpoint=use_checkpoint, use_swin=use_swin)
      self.upsample = ocnn.modules.OctreeDeconvGnRelu(
          num_embed, num_embed, group=32, kernel_size=[2], stride=2)

    self.mask_ratio_generator = stats.truncnorm(
        (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

    self.apply(self._init_weights)  # initialize weights
    logger.info("number of parameters: %e", sum(p.numel()
                for p in self.parameters()))

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

  def random_masking(self, seq_len, orders, mask_rate=None):
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

  def forward_blocks(self, x, octree, blocks, depth_list):
    _, depth2batch_indices = get_depth2batch_indices(
        octree, depth_list)
    x = depth2batch(x, depth2batch_indices)
    x = blocks(x, octree, depth_list)
    x = batch2depth(x, depth2batch_indices)
    return x

  def forward_model(self, x, octree, depth_low, depth_high, nnum_split):
    # if only split signals, not use vq sample
    apply_vq_sample = nnum_split < x.shape[0]
    if self.use_vq_sample and apply_vq_sample:
      depth_list_main = list(range(depth_low, depth_high)) + [depth_high - 1]
    else:
      depth_list_main = list(range(depth_low, depth_high + 1))

    if self.use_vq_sample:
      depth_list_vq = list(range(depth_low, depth_high + 1))
      x = self.forward_blocks(x, octree, self.vq_encoder, depth_list_vq)
      if apply_vq_sample:
        x_split = x[:nnum_split]
        x_vq = x[nnum_split:]
        x_vq = self.downsample(x_vq, octree, depth_high)
        x = torch.cat([x_split, x_vq], dim=0)

    x = self.forward_blocks(
        x, octree, self.blocks, depth_list_main)

    if self.use_vq_sample:
      if apply_vq_sample:
        # skip connection
        x_split = x[:nnum_split]
        x_vq = x[nnum_split:] + x_vq
        x_vq = self.upsample(x_vq, octree, depth_high - 1)
        x = torch.cat([x_split, x_vq], dim=0)
      x = self.forward_blocks(x, octree, self.vq_decoder, depth_list_vq)
    x = self.ln_x(x)
    return x

  def forward(self, octree_in, depth_low, depth_high, category=None, split=None, vqvae=None):
    x_token_embeddings = torch.empty(
        (0, self.num_embed), device=octree_in.device)
    batch_size = octree_in.batch_size

    if category == None:
      category = torch.zeros(batch_size).long().to(octree_in.device)
    cond = self.class_emb(category)  # 1 x C

    x_token_embeddings = torch.cat(
        [x_token_embeddings, self.split_emb(split)])  # S x C
    targets_split = copy.deepcopy(split)
    nnum_split = x_token_embeddings.shape[0]

    with torch.no_grad():
      vq_code = vqvae.extract_code(octree_in)
      # quantizer dose not affect by the order
      zq, indices, _ = vqvae.quantizer(vq_code)
      targets_vq = copy.deepcopy(indices)
    zq = self.vq_proj(zq)
    x_token_embeddings = torch.cat([x_token_embeddings, zq], dim=0)

    batch_id = get_batch_id(octree_in, range(depth_low, depth_high + 1))
    seq_len = x_token_embeddings.shape[0]
    orders = torch.randperm(seq_len, device=x_token_embeddings.device)
    mask = self.random_masking(seq_len, orders).bool()
    batch_cond = cond[batch_id]
    x_token_embeddings = torch.where(
        mask.unsqueeze(1), batch_cond, x_token_embeddings)

    x = self.forward_model(
        x_token_embeddings, octree_in, depth_low, depth_high, nnum_split)

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
    mask_vq = mask[nnum_split:]
    vq_logits = self.vq_head(x[nnum_split:])
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
  def generate(self, octree, depth_low, depth_high, token_embeddings=None, category=None, vqvae=None):
    batch_size = octree.batch_size
    if category == None:
      category = torch.zeros(batch_size).long().to(octree.device)
    cond = self.class_emb(category)  # 1 x C

    if token_embeddings is None:
      token_embeddings = torch.empty(
          (0, self.num_embed), device=octree.device)

    vq_code = None
    for d in range(depth_low, depth_high + 1):
      nnum_d = octree.nnum[d]
      nnum_split = sum([octree.nnum[i]
                       for i in range(depth_low, min(d + 1, depth_high))])

      mask = torch.ones(nnum_d, device=octree.device).bool()
      if d < depth_high:
        split = -1 * torch.ones(nnum_d, device=octree.device).long()
      else:
        vq_indices = -1 * \
            torch.ones((nnum_d, self.vq_groups), device=octree.device).long()
        vq_code = torch.zeros(nnum_d, self.num_vq_embed, device=octree.device)
      # fullly masked
      batch_id = get_batch_id(
          octree, range(depth_low, d + 1))
      token_embedding_d = cond[batch_id][token_embeddings.shape[0]:]
      orders = torch.randperm(nnum_d, device=octree.device)

      # set generate parameters
      num_iters = self.num_iters[d - depth_low] \
          if isinstance(self.num_iters, list) else self.num_iters
      start_temperature = self.start_temperature[d - depth_low] \
          if isinstance(self.start_temperature, list) else self.start_temperature

      for i in tqdm(range(num_iters)):
        x = torch.cat([token_embeddings, token_embedding_d], dim=0)
        x = self.forward_model(x, octree, depth_low, d, nnum_split)
        x = x[-nnum_d:, :]

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_iters)
        mask_len = torch.Tensor([np.floor(nnum_d * mask_ratio)]).cuda()

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(
            torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)).long()

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = self.mask_by_order(mask_len, orders).bool()

        if i >= num_iters - 1:
          mask_to_pred = mask.bool()
        else:
          mask_to_pred = torch.logical_xor(
              mask.bool(), mask_next.bool())
        mask = mask_next

        temperature = start_temperature * \
            ((num_iters - i) / num_iters)

        if d < depth_high:
          split_logits = self.split_head(x)
          # remask tokens that have poor confidence
          if i > num_iters * self.remask_stage:
            remask = self.get_remask(split_logits, split, mask)
            mask_to_pred = mask_to_pred | remask

          ix = sample(split_logits[mask_to_pred], temperature=temperature)
          split[mask_to_pred] = ix
          token_embedding_d[mask_to_pred] = self.split_emb(ix)
        else:
          vq_logits = self.vq_head(x)
          # remask tokens that have poor confidence
          if i > num_iters * self.remask_stage:
            vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)
            remask = self.get_remask(vq_logits, vq_indices, mask, topk=5)
            mask_to_pred = mask_to_pred | remask
          vq_logits = vq_logits[mask_to_pred].reshape(-1, self.vq_size)
          ix = sample(vq_logits, temperature=temperature)
          ix = ix.reshape(-1, self.vq_groups)
          vq_indices[mask_to_pred] = ix
          with torch.no_grad():
            zq = vqvae.quantizer.extract_code(ix)
            vq_code[mask_to_pred] = zq
            token_embedding_d[mask_to_pred] = self.vq_proj(zq)

      token_embeddings = torch.cat(
          [token_embeddings, token_embedding_d], dim=0)
      if d < depth_high:
        split = split.long()
        octree = seq2octree(octree, split, d, d + 1)
        # export_octree(
        #     octree, d + 1, f"mytools/octree/depth{d+1}/", index=get_rank())

    return octree, vq_code
