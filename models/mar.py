import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.octformer import OctFormer, SinPosEmb, OctreeConvPosEmb
from models.vae import DiagonalGaussian
from utils.utils import seq2octree, sample
logger = logging.getLogger(__name__)


class MAR(nn.Module):
  """ Masked Autoencoder with VisionTransformer backbone
  """

  def __init__(self,
               num_embed=256,
               num_vq_embed=256,
               num_heads=8,
               num_blocks=8,
               num_classes=1,
               split_size=2,
               vq_size=128,
               patch_size=4096,
               dilation=2,
               drop_rate=0.1,
               pos_emb_type="SinPosEmb",
               use_checkpoint=True,
               vae_name="vqvae",
               mask_ratio_min=0.7,
               start_temperature=1.0,
               num_iters=256,
               **kwargs):
    super(MAR, self).__init__()
    self.num_embed = num_embed
    self.num_vq_embed = num_vq_embed
    self.num_blocks = num_blocks
    self.start_temperature = start_temperature
    self.num_iters = num_iters
    self.vq_name = vae_name

    # self.pos_emb = eval(pos_emb_type)(num_embed)

    self.split_emb = nn.Embedding(split_size, num_embed)
    self.class_emb = nn.Embedding(num_classes, num_embed)
    self.vq_proj = nn.Linear(num_vq_embed, num_embed)

    self.blocks = OctFormer(
        channels=num_embed, num_blocks=num_blocks, num_heads=num_heads,
        patch_size=patch_size, dilation=dilation, attn_drop=drop_rate,
        proj_drop=drop_rate, pos_emb=eval(pos_emb_type), nempty=False,
        use_checkpoint=use_checkpoint)

    self.ln_x = nn.LayerNorm(num_embed)
    self.split_head = nn.Linear(num_embed, split_size)
    if self.vq_name == "vqvae":
      self.vq_head = nn.Linear(num_embed, vq_size)
    elif self.vq_name == "vae":
      self.vq_head = nn.Linear(num_embed, num_vq_embed)

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
    num_masked_tokens = max(int(np.ceil(seq_len * mask_rate)), 1)
    return self.mask_by_order(num_masked_tokens, orders)

  def mask_by_order(self, mask_len, orders):
    mask = torch.zeros(orders.shape[0], device=orders.device).long()
    mask[orders[:mask_len]] = 1
    return mask

  def get_depth_index(self, octree, depth_low, depth_high):
    return torch.cat([torch.ones(octree.nnum[d], device=octree.device).long() * d
                      for d in range(depth_low, depth_high + 1)])

  def forward(self, octree_in, depth_low, depth_high, category=None, split=None, vqvae=None):
    x_token_embeddings = torch.empty((0, self.num_embed), device=octree_in.device)
    batch_size = octree_in.batch_size

    if category == None:
      category = torch.zeros(batch_size).long().to(octree_in.device)
    cond = self.class_emb(category)  # 1 x C


    if split is not None:
      x_token_embeddings = torch.cat([x_token_embeddings, self.split_emb(split)])  # S x C
      targets_split = copy.deepcopy(split)
      nnum_split = x_token_embeddings.shape[0]
    else:
      nnum_split = 0

    if vqvae is not None:
      with torch.no_grad():
        vq_code = vqvae.extract_code(octree_in)
        if self.vq_name == "vqvae":
          zq, indices, _ = vqvae.quantizer(vq_code)
          targets_vq = copy.deepcopy(indices)
        elif self.vq_name == "vae":
          posterior = DiagonalGaussian(vq_code, kl_std=0.25)
          zq = posterior.sample()
          targets_vq = copy.deepcopy(zq)
      zq = self.vq_proj(zq)
      x_token_embeddings = torch.cat([x_token_embeddings, zq], dim=0)

    seq_len = x_token_embeddings.shape[0]
    orders = torch.randperm(seq_len, device=x_token_embeddings.device)
    mask = self.random_masking(seq_len, orders).bool()
    x_token_embeddings[mask] = cond
    # positional embedding
    # position_embeddings = self.pos_emb(
    #     x_token_embeddings, octree_in)  # S x C
    # x = x_token_embeddings + position_embeddings[:seq_len]

    # get depth index
    depth_idx = self.get_depth_index(octree_in, depth_low, depth_high)

    x, presents = self.blocks(x_token_embeddings, octree_in, depth_low,
                              depth_high, past=None, group_idx=depth_idx)
    x = self.ln_x(x)

    output = {}
    if split is not None:
      mask_split = mask[:nnum_split]
      split_logits = self.split_head(x[:nnum_split])
      output['split_loss'] = F.cross_entropy(
          split_logits[mask_split], targets_split[mask_split])
    else:
      output['split_loss'] = torch.tensor(0.0).to(octree_in.device)

    if vqvae is not None:
      mask_vq = mask[nnum_split:]
      vq_logits = self.vq_head(x[nnum_split:])
      if self.vq_name == "vqvae":
        output['vq_loss'] = F.cross_entropy(
            vq_logits[mask_vq], targets_vq[mask_vq])
        # Top-k Accuracy
        with torch.no_grad():
          mask_vq = mask[nnum_split:]
          top5 = torch.topk(vq_logits[mask_vq], 5, dim=1).indices
          correct_top5 = top5.eq(targets_vq[mask_vq].view(-1, 1).expand_as(top5))
          output['top5_accuracy'] = correct_top5.sum().float() / mask_vq.sum().float()
      elif self.vq_name == "vae":
        output['vq_loss'] = F.mse_loss(vq_logits[mask_vq], targets_vq[mask_vq])
    else:
      output['vq_loss'] = torch.tensor(0.0).to(split.device)

    return output

  @torch.no_grad()
  def generate(self, octree, depth_low, depth_high, token_embeddings=None, category=None, vqvae=None):
    if category == None:
      category = torch.zeros(1).long().to(octree.device)
    cond = self.class_emb(category)  # 1 x C

    if token_embeddings is None:
      token_embeddings = torch.empty(
          (0, self.num_embed), device=octree.device)

    vq_code = None
    # past = torch.empty(
    # (self.n_layer, 0, self.n_embed * 3), device=octree.device)
    past = None
    for d in range(depth_low, depth_high + 1):
      # if not need to generate vq code
      if d == depth_high and vqvae == None:
        break
      
      start_temperature = self.start_temperature * (0.8 ** (d - depth_low))
      # get depth index
      depth_idx = self.get_depth_index(octree, depth_low, d)
      nnum_d = octree.nnum[d]

      mask = torch.ones(nnum_d, device=octree.device).bool()
      if d < depth_high:
        split = -1 * torch.ones(nnum_d, device=octree.device).long()
      else:
        vq_code = torch.zeros(nnum_d, self.num_vq_embed, device=octree.device)
      token_embedding_d = cond.repeat(nnum_d, 1)
      orders = torch.randperm(nnum_d, device=octree.device)

      for i in tqdm(range(self.num_iters)):
        x = torch.cat([token_embeddings, token_embedding_d], dim=0)
        # position_embeddings = self.pos_emb(
        #     x, octree, depth_low, d)  # S x C
        # x = x + position_embeddings[:x.shape[0], :]
        x, _ = self.blocks(x, octree, depth_low, d,
                           group_idx=depth_idx)  # B x S x C
        x = x[-nnum_d:, :]
        x = self.ln_x(x)

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (i + 1) / self.num_iters)
        mask_len = torch.Tensor([np.floor(nnum_d * mask_ratio)]).cuda()

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(
            torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)).long()

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = self.mask_by_order(mask_len, orders).bool()

        if i >= self.num_iters - 1:
          mask_to_pred = mask.bool()
        else:
          mask_to_pred = torch.logical_xor(
              mask.bool(), mask_next.bool())
        mask = mask_next

        temperature = start_temperature * \
            ((self.num_iters - i) / self.num_iters)

        if d < depth_high:
          split_logits = self.split_head(x[mask_to_pred])
          ix = sample(split_logits, temperature=temperature)
          split[mask_to_pred] = ix
          token_embedding_d[mask_to_pred] = self.split_emb(ix)
        else:
          vq_logits = self.vq_head(x[mask_to_pred])
          if self.vq_name == "vqvae":
            ix = sample(vq_logits, top_k=5, temperature=temperature)
            with torch.no_grad():
              zq = vqvae.quantizer.embedding(ix)
              vq_code[mask_to_pred] = zq
              token_embedding_d[mask_to_pred] = self.vq_proj(zq)
          elif self.vq_name == "vae":
            vq_code[mask_to_pred] = vq_logits
            token_embedding_d[mask_to_pred] = self.vq_proj(vq_logits)

      token_embeddings = torch.cat(
          [token_embeddings, token_embedding_d], dim=0)
      if d < depth_high:
        octree = seq2octree(octree, split, d, d + 1)
        # utils.export_octree(
        # octree, d + 1, f"mytools/octree/depth{d+1}/", index=get_rank())

    return octree, vq_code
