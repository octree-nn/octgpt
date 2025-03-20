import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import logging
import copy
import ocnn
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from models.octformer import OctFormer, OctreeT
from models.positional_embedding import SinPosEmb, AbsPosEmb, RMSNorm
from models.vae import DiagonalGaussian
from utils.utils import seq2octree, sample, depth2batch, batch2depth, \
    get_depth2batch_indices, get_batch_id, export_octree
from utils.distributed import get_rank
logger = logging.getLogger(__name__)


class OctGPT(nn.Module):

  def __init__(self,
               num_embed=256,
               num_heads=8,
               num_blocks=8,
               num_classes=1,
               patch_size=4096,
               dilation=2,
               buffer_size=64,
               drop_rate=0.1,
               pos_emb_type="AbsPosEmb",
               norm_type="RMSNorm",
               use_checkpoint=True,
               random_flip=0.0,
               mask_ratio_min=0.7,
               start_temperature=1.0,
               remask_stage=0.9,
               num_iters=[64, 128, 128, 256],
               vqvae_config=None,
               condition_type="none",
               condition_policy="concat",
               context_dim=512,
               **kwargs):
    super(OctGPT, self).__init__()
    self.vqvae_config = vqvae_config
    self.num_embed = num_embed
    self.num_heads = num_heads
    self.num_blocks = num_blocks
    self.patch_size = patch_size
    self.dilation = dilation
    self.buffer_size = buffer_size
    self.drop_rate = drop_rate
    self.pos_emb_type = pos_emb_type
    self.norm_type = norm_type
    self.use_checkpoint = use_checkpoint
    self.random_flip = random_flip
    self.start_temperature = start_temperature
    self.remask_stage = remask_stage
    self.num_iters = num_iters
    self.condition_type = condition_type
    self.condition_policy = condition_policy
    self.context_dim = context_dim
    self.split_size = 2  # 0/1 indicates the split signals
    self.num_vq_embed = vqvae_config.embedding_channels

    if vqvae_config.quantizer_type == "bsq":
      self.vq_size = 2
      self.vq_groups = vqvae_config.embedding_channels
    else:
      raise ValueError("Unsupported quantizer type")

    self.split_emb = nn.Embedding(self.split_size, num_embed)
    self.class_emb = nn.Embedding(num_classes, num_embed)
    self.vq_proj = nn.Linear(self.num_vq_embed, num_embed)

    if condition_type in ["image", "text"]:
      self.mask_token = nn.Parameter(torch.zeros(1, num_embed))
      self.ln_cond = eval(self.norm_type)(context_dim)
      self.cond_proj = nn.Linear(context_dim, num_embed)

    # Initialize the blocks
    self._init_blocks()

    self.split_head = nn.Linear(num_embed, self.split_size)
    self.vq_head = nn.Linear(num_embed, self.vq_size * self.vq_groups)

    self.mask_ratio_generator = stats.truncnorm(
        (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

    self.apply(self._init_weights)  # initialize weights
    if condition_type not in ["none", "category"]:
      self._init_weights(self.mask_token)

  def _init_blocks(self):
    self.encoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        nempty=False, use_checkpoint=self.use_checkpoint,
        use_ctx=self.condition_policy == "cross_attn", ctx_dim=self.context_dim, ctx_interval=2,
        pos_emb=eval(self.pos_emb_type), norm_layer=eval(self.norm_type))
    self.encoder_ln = eval(self.norm_type)(self.num_embed)

    self.decoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        nempty=False, use_checkpoint=self.use_checkpoint,
        use_ctx=self.condition_policy=="cross_attn", ctx_dim=self.context_dim, ctx_interval=2,
        pos_emb=eval(self.pos_emb_type), norm_layer=eval(self.norm_type))
    self.decoder_ln = eval(self.norm_type)(self.num_embed)

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
    num_masked_tokens = max(int(np.ceil(seq_len * mask_rate)), 1)
    return self.mask_by_order(num_masked_tokens, orders)

  def mask_by_order(self, mask_len, orders):
    mask = torch.zeros(orders.shape[0], device=orders.device).long()
    mask[orders[:mask_len]] = 1
    return mask

  def random_masking(self, x, mask, octree, depth_list, cond=None):
    if self.condition_type in ["none", "category"]:
      batch_id = get_batch_id(octree, depth_list)
      mask_tokens = cond[batch_id]
    elif self.condition_type in ['image', 'text']:
      mask_tokens = self.mask_token.repeat(x.shape[0], 1)
    x = torch.where(mask.bool().unsqueeze(1), mask_tokens, x)
    return x

  def add_buffer(self, x, mask, cond):
    batch_size = cond.shape[0]
    # Add condition to the input
    if len(cond.shape) == 2:  # For None, category
      buffer = cond.reshape(batch_size, 1, -1)
      buffer = buffer.repeat(1, self.buffer_size, 1).reshape(-1, self.num_embed)
    elif len(cond.shape) == 3:  # For ViT, DINOv2, CLIP
      buffer = self.cond_proj(self.ln_cond(cond))
      if buffer.shape[1] < self.buffer_size:
        buffer = torch.cat([
            buffer, torch.zeros(batch_size, self.buffer_size - buffer.shape[1], self.num_embed, device=x.device)], dim=1)
      buffer = buffer.reshape(-1, self.num_embed)

    mask_buffer = torch.zeros(buffer.shape[0], device=x.device).bool()
    x = torch.cat([buffer, x], dim=0)
    mask = torch.cat([mask_buffer, mask], dim=0)
    return x, mask

  def forward_blocks(self, x, octree: OctreeT, blocks, context: torch.Tensor):
    x = depth2batch(x, octree.indices)
    x = blocks(x, octree, context)
    x = batch2depth(x, octree.indices)
    return x

  def forward_model(self, x, octree, depth_low, depth_high, mask, nnum_split, cond=None):
    batch_size = octree.batch_size
    depth_list = list(range(depth_low, depth_high + 1))
    x, mask = self.add_buffer(x, mask, cond)
    if self.condition_type in ["image", "text"] and self.condition_policy == "cross_attn":
      context = cond
    else:
      context = None

    x_enc = x.clone()
    x_enc = x_enc[~mask]
    octreeT_encoder = OctreeT(
        octree, x_enc.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list, data_mask=mask, buffer_size=self.buffer_size)
    x_enc = self.forward_blocks(x_enc, octreeT_encoder, self.encoder, context)
    x_enc = self.encoder_ln(x_enc)
    x[~mask] = x_enc

    octreeT_decoder = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list, buffer_size=self.buffer_size)
    x = self.forward_blocks(x, octreeT_decoder, self.decoder, context)
    x = self.decoder_ln(x)
    x = x[batch_size * self.buffer_size:]

    return x

  def forward(self, octree_in, depth_low, depth_high, condition=None, split=None, vqvae=None):
    batch_size = octree_in.batch_size

    if self.condition_type == "none":
      condition = torch.zeros(batch_size).long().to(octree_in.device)
      cond = self.class_emb(condition)  # 1 x C
    elif self.condition_type == "category":
      cond = self.class_emb(condition)
    elif self.condition_type in ['image', 'text']:
      cond = condition

    targets_split = split.clone().detach()
    split_token_embeddings = self.split_emb(split)  # (nnum_split, C)
    nnum_split = split_token_embeddings.shape[0]

    with torch.no_grad():
      vq_code = vqvae.extract_code(octree_in)
      zq, indices, _ = vqvae.quantizer(vq_code)
      targets_vq = copy.deepcopy(indices)
      if self.random_flip > 0.0 and self.training:
        flip = torch.rand_like(indices.float()) < self.random_flip
        indices = torch.where(flip, 1 - indices, indices)
        zq = vqvae.quantizer.extract_code(indices)
    vq_token_embeddings = self.vq_proj(zq)
    nnum_vq = vq_token_embeddings.shape[0]

    # x_token_embeddings: (nnum, C)
    x_token_embeddings = torch.cat(
        [split_token_embeddings, vq_token_embeddings], dim=0)

    seq_len = x_token_embeddings.shape[0]
    orders = torch.randperm(seq_len, device=x_token_embeddings.device)
    mask = self.get_mask(seq_len, orders).bool()
    x_token_embeddings = self.random_masking(
        x_token_embeddings, mask, octree_in, list(range(depth_low, depth_high + 1)), cond=cond)

    # forward model
    x = self.forward_model(
        x_token_embeddings, octree_in, depth_low, depth_high, mask, nnum_split, cond=cond)

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
    if self.random_flip > 0.0:
      output['vq_loss'] = F.cross_entropy(
          vq_logits.reshape(-1, self.vq_size), targets_vq.reshape(-1))
    else:
      output['vq_loss'] = F.cross_entropy(
          vq_logits[mask_vq].reshape(-1, self.vq_size), targets_vq[mask_vq].reshape(-1))
    # Top-k Accuracy
    with torch.no_grad():
      correct_top5 = self.get_correct_topk(
          vq_logits[mask_vq].reshape(-1, self.vq_size),
          targets_vq[mask_vq].reshape(-1),
          topk=5)
      output['top5_accuracy'] = correct_top5.sum().float() / \
          (mask_vq.sum().float() * self.vq_groups)
    return output

  def get_correct_topk(self, logits, targets, topk=1):
    topk = min(topk, logits.shape[-1] - 1)
    topk = torch.topk(logits, topk, dim=-1).indices
    correct_topk = topk.eq(targets.unsqueeze(-1).expand_as(topk))
    return correct_topk

  def get_remask(self, logits, tokens, mask, remask_prob=0.2, topk=1):
    correct_topk = self.get_correct_topk(logits, tokens, topk=topk)
    correct_by_group = correct_topk.any(dim=-1)
    remask = torch.zeros_like(mask).bool()

    if len(correct_by_group.shape) == 1:  # for split
      num_incorrect = (~correct_by_group).long()
      remask_scores = -logits[torch.arange(logits.shape[0]), tokens]
    else:  # for vq
      num_incorrect = (~correct_by_group).sum(dim=-1)
      remask_scores = num_incorrect
    num_incorrect[mask] = 0
    num_remask = int(num_incorrect.bool().sum() * remask_prob)
    remask_indices = torch.topk(remask_scores, num_remask).indices
    remask[remask_indices] = True
    remask = remask & ~mask
    return remask

  @torch.no_grad()
  def generate(self, octree, depth_low, depth_high, token_embeddings=None, condition=None, vqvae=None):
    batch_size = octree.batch_size
    if self.condition_type == "none":
      condition = torch.zeros(batch_size).long().to(octree.device)
      cond = self.class_emb(condition)  # 1 x C
    elif self.condition_type == "category":
      cond = self.class_emb(condition)
    elif self.condition_type in ['image', 'text']:
      cond = condition

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
          token_embedding_d, mask_d, octree, [d], cond=cond)
      orders = torch.randperm(nnum_d, device=octree.device)

      # set generate parameters
      num_iters = self.num_iters[d - depth_low] \
          if isinstance(self.num_iters, list) else self.num_iters
      start_temperature = self.start_temperature[d - depth_low] \
          if isinstance(self.start_temperature, list) else self.start_temperature

      for i in tqdm(range(num_iters)):
        x = torch.cat([token_embeddings, token_embedding_d], dim=0)
        x = self.forward_model(x, octree, depth_low, d, nnum_split=nnum_split,
                               mask=torch.cat([mask, mask_d]), cond=cond)
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

        if d < depth_high:
          split_logits = self.split_head(x)
          # remask tokens that have poor confidence
          if i > num_iters * self.remask_stage:
            remask = self.get_remask(
                split_logits, split_d, mask_d, remask_prob=0.2)
            mask_to_pred = mask_to_pred | remask
          ix = sample(split_logits[mask_to_pred], temperature=temperature)
          split_d[mask_to_pred] = ix
          token_embedding_d[mask_to_pred] = self.split_emb(ix)
        else:
          vq_logits = self.vq_head(x)
          if i > num_iters * self.remask_stage:
            vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)
            remask = self.get_remask(vq_logits, vq_indices_d, mask_d, topk=5, remask_prob=0.1)
            mask_to_pred = mask_to_pred | remask
          vq_logits = vq_logits[mask_to_pred].reshape(-1, self.vq_size)
          ix = sample(vq_logits, temperature=temperature)
          ix = ix.reshape(-1, self.vq_groups)
          vq_indices_d[mask_to_pred] = ix.long()
          with torch.no_grad():
            zq = vqvae.quantizer.extract_code(ix)
            vq_code_d[mask_to_pred] = zq.float()
            token_embedding_d[mask_to_pred] = self.vq_proj(zq).float()

      token_embeddings = torch.cat(
          [token_embeddings, token_embedding_d], dim=0)
      if d < depth_high:
        split_d = split_d.long()
        octree = seq2octree(octree, split_d, d, d + 1)
        # export_octree(
        #   octree, d + 1, f"mytools/octree/depth{d+1}/", index=get_rank())
      else:
        vq_code = torch.cat([vq_code, vq_code_d], dim=0)

    return octree, vq_code