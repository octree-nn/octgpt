# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import ocnn
from ocnn.octree import Octree
from typing import Optional
from torch.utils.checkpoint import checkpoint
from models.positional_embedding import RotaryPosEmb, SinPosEmb, FULL_DEPTH, MAX_DEPTH
from utils.utils import get_depth2batch_indices, depth2batch, batch2depth


class OctreeT(Octree):

  def __init__(self, octree: Octree, data_length: int, patch_size: int = 24,
               dilation: int = 4, nempty: bool = True, use_swin: bool = True,
               depth_list: list = None, **kwargs):
    super().__init__(octree.depth, octree.full_depth)
    self.__dict__.update(octree.__dict__)

    self.patch_size = patch_size
    self.dilation = dilation    # TODO dilation as a list
    self.nempty = nempty
    self.use_swin = use_swin
    self.depth_list = depth_list if depth_list != None else \
        list(range(self.full_depth, self.depth + 1))
    self.invalid_mask_value = -1e3

    self.block_num = patch_size * dilation

    self.nnum_t = torch.tensor(data_length, device=self.device)
    self.nnum_a = (torch.ceil(self.nnum_t / self.block_num)
                   * self.block_num).int()
    if self.use_swin:
      self.swin_nnum_pad = self.patch_size // 2
      self.swin_nnum_a = (torch.ceil(
          (self.nnum_t + self.swin_nnum_pad) / self.block_num) * self.block_num).int()
    self.batch_idx, self.indices = get_depth2batch_indices(
        self, self.depth_list)
    self.build_t()

  def build_t(self):
    self.depth_idx = self.build_depth_idx()
    self.patch_mask, self.dilate_mask = self.build_attn_mask()
    self.patch_tf_mask, self.dilate_tf_mask = self.build_teacher_forcing_mask()
    self.xyz = self.build_xyz()
    if self.use_swin:
      self.swin_patch_mask, self.swin_dilate_mask = self.build_attn_mask(
          use_swin=True)
      self.swin_patch_tf_mask, self.swin_dilate_tf_mask = \
          self.build_teacher_forcing_mask(use_swin=True)

  def build_batch_idx(self, use_swin=False):
    batch_idx_patch = self.patch_partition(
        self.batch_idx[:self.nnum_t], self.batch_size, use_swin=use_swin)
    return batch_idx_patch

  def build_depth_idx(self):
    depth_idx = torch.cat([torch.ones(self.nnum[self.depth_list[i]], device=self.device).long() * i
                           for i in range(len(self.depth_list))])
    depth_idx = depth2batch(depth_idx, self.indices)
    return depth_idx

  def build_attn_mask(self, use_swin=False):
    batch = self.build_batch_idx(use_swin)
    mask = batch.view(-1, self.patch_size)
    patch_mask = self._calc_attn_mask(mask)

    mask = batch.view(-1, self.patch_size, self.dilation)
    mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
    dilate_mask = self._calc_attn_mask(mask)
    return patch_mask, dilate_mask

  def build_teacher_forcing_mask(self, use_swin=False):
    max_value = self.depth_idx[:self.nnum_t].max()
    group = self.patch_partition(
        self.depth_idx[:self.nnum_t], fill_value=max_value + 1, use_swin=use_swin)
    mask = group.view(-1, self.patch_size)
    patch_tf_mask = self._calc_attn_mask(mask, cond="le")

    mask = group.view(-1, self.patch_size, self.dilation)
    mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
    dilate_tf_mask = self._calc_attn_mask(mask, cond="le")
    return patch_tf_mask, dilate_tf_mask

  def _calc_attn_mask(self, mask: torch.Tensor, cond="neq"):
    attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
    if cond == "neq":
      mask_label = attn_mask != 0
    elif cond == "le":
      mask_label = attn_mask < 0
    else:
      raise ValueError("Invalid condition")
    attn_mask = torch.zeros_like(attn_mask).masked_fill(
        mask_label, self.invalid_mask_value)
    return attn_mask

  def build_xyz(self):
    max_scale = 2 ** (MAX_DEPTH + 1)

    def rescale_pos(x, scale):
      x = x * max_scale // scale
      x += max_scale // scale // 2
      return x

    xyz = []
    for d in self.depth_list:
      scale = 2 ** d
      x, y, z, b = self.xyzb(d)
      x = rescale_pos(x, scale)
      y = rescale_pos(y, scale)
      z = rescale_pos(z, scale)
      xyz.append(torch.stack([x, y, z], dim=1))
    xyz = torch.cat(xyz, dim=0).float()
    xyz = depth2batch(xyz, self.indices)
    return xyz

  def patch_partition(self, data: torch.Tensor, fill_value=0, use_swin=False):
    assert data.shape[0] == self.nnum_t

    if use_swin:
      head = data.new_full((self.swin_nnum_pad,) + data.shape[1:], fill_value)
      num = self.swin_nnum_a - self.nnum_t - self.swin_nnum_pad
      tail = data.new_full((num,) + data.shape[1:], fill_value)
      data = torch.cat([head, data, tail], dim=0)
    else:
      num = self.nnum_a - self.nnum_t
      tail = data.new_full((num,) + data.shape[1:], fill_value)
      data = torch.cat([data, tail], dim=0)

    return data

  def patch_reverse(self, data: torch.Tensor, use_swin=False):
    if use_swin:
      data = data[self.swin_nnum_pad:self.nnum_t + self.swin_nnum_pad]
    else:
      data = data[:self.nnum_t]
    return data


class MLP(torch.nn.Module):

  def __init__(self, in_features: int, hidden_features: Optional[int] = None,
               out_features: Optional[int] = None, activation=torch.nn.GELU,
               drop: float = 0.0, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features or in_features
    self.hidden_features = hidden_features or in_features

    self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
    self.act = activation()
    self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
    self.drop = torch.nn.Dropout(drop, inplace=True)

  def forward(self, data: torch.Tensor):
    data = self.fc1(data)
    data = self.act(data)
    data = self.drop(data)
    data = self.fc2(data)
    data = self.drop(data)
    return data


class RPE(torch.nn.Module):

  def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
    super().__init__()
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.dilation = dilation
    self.pos_bnd = self.get_pos_bnd(patch_size)
    self.rpe_num = 2 * self.pos_bnd + 1
    self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
    torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

  def get_pos_bnd(self, patch_size: int):
    return int(0.8 * patch_size * self.dilation**0.5)

  def xyz2idx(self, xyz: torch.Tensor):
    mul = torch.arange(3, device=xyz.device) * self.rpe_num
    xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
    idx = xyz + (self.pos_bnd + mul)
    return idx

  def forward(self, xyz):
    idx = self.xyz2idx(xyz)
    out = self.rpe_table.index_select(0, idx.reshape(-1))
    out = out.view(idx.shape + (-1,)).sum(3)
    out = out.permute(0, 3, 1, 2)    # (N, K, K, H) -> (N, H, K, K)
    return out

  def extra_repr(self) -> str:
    return 'num_heads={}, pos_bnd={}, dilation={}'.format(
        self.num_heads, self.pos_bnd, self.dilation)    # noqa


class OctreeAttention(torch.nn.Module):

  def __init__(self, dim: int, patch_size: int, num_heads: int,
               qkv_bias: bool = True, qk_scale: Optional[float] = None,
               attn_drop: float = 0.0, proj_drop: float = 0.0,
               dilation: int = 1, use_rpe: bool = False, use_flash: bool = True,
               use_rope: bool = True, use_swin: bool = True, **kwargs):
    super().__init__()
    self.dim = dim
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.dilation = dilation
    self.use_rpe = use_rpe
    self.use_flash = use_flash
    self.use_rope = use_rope
    self.use_swin = use_swin
    self.scale = qk_scale or (dim // num_heads) ** -0.5

    self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
    if self.use_flash:
      self.attn_drop = attn_drop
    else:
      self.attn_drop = torch.nn.Dropout(attn_drop)
    self.proj = torch.nn.Linear(dim, dim)
    self.proj_drop = torch.nn.Dropout(proj_drop)
    self.softmax = torch.nn.Softmax(dim=-1)

    # NOTE: self.rpe is not used in the original experiments of my paper. When
    # releasing the code, I added self.rpe because I observed that it could
    # stablize the training process and improve the performance on ScanNet by
    # 0.3 to 0.5; on the other datasets, the improvements are more marginal. So
    # it is not indispensible, and can be removed by setting `use_rpe` as False.
    self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None
    if self.use_rope:
      self.rope = RotaryPosEmb(dim=dim, num_heads=num_heads, rope_mixed=True)

  def forward(self, data: torch.Tensor, octree: OctreeT):
    H = self.num_heads
    K = self.patch_size
    C = self.dim
    D = self.dilation

    if D > 1:
      mask = octree.dilate_mask if not self.use_swin else octree.swin_dilate_mask
      mask += octree.dilate_tf_mask if not self.use_swin else octree.swin_dilate_tf_mask
    else:
      mask = octree.patch_mask if not self.use_swin else octree.swin_patch_mask
      mask += octree.patch_tf_mask if not self.use_swin else octree.swin_patch_tf_mask
    mask = mask.float()

    def patchify_qkv(qkv: torch.Tensor):
      # patch partition
      qkv = octree.patch_partition(qkv, use_swin=self.use_swin)
      if D > 1:    # dilation
        qkv = qkv.view(-1, K, D, C * 3).transpose(1, 2).reshape(-1, C * 3)
      qkv = qkv.view(-1, K, C * 3)
      qkv = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]
      return q, k, v

    # qkv
    qkv = self.qkv(data)

    # apply rotary position embedding
    if self.use_rope:
      qkv = self.rope(qkv, octree)

    Q = K
    q, k, v = patchify_qkv(qkv)

    if self.use_flash:
      data = F.scaled_dot_product_attention(
          q, k, v, attn_mask=mask.unsqueeze(1),
          dropout_p=self.attn_drop if self.training else 0.0,
          scale=self.scale)
      data = data.transpose(1, 2).reshape(-1, C)
    else:
      # attn
      attn = q @ k.transpose(-2, -1) * self.scale  # (N, H, K, K)
      # attn = self.apply_rpe(attn, rel_pos)    # (N, H, K, K)
      attn = attn + mask.unsqueeze(1)
      attn = self.softmax(attn)
      attn = self.attn_drop(attn)
      data = (attn @ v).transpose(1, 2).reshape(-1, C)

    # patch reverse
    if D > 1:    # dilation
      data = data.view(-1, D, Q, C).transpose(1, 2).reshape(-1, C)

    # patch reverse
    data = octree.patch_reverse(data, use_swin=self.use_swin)

    # ffn
    data = self.proj(data)
    data = self.proj_drop(data)
    return data

  def apply_rpe(self, attn, rel_pos):
    if self.use_rpe:
      attn = attn + self.rpe(rel_pos)
    return attn

  def extra_repr(self) -> str:
    return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
        self.dim, self.patch_size, self.num_heads, self.dilation)    # noqa


class OctFormerBlock(torch.nn.Module):

  def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
               dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
               qk_scale: Optional[float] = None, attn_drop: float = 0.0,
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               use_swin: bool = False, pos_emb: torch.nn.Module = SinPosEmb,
               activation: torch.nn.Module = torch.nn.GELU,
               **kwargs):
    super().__init__()
    self.norm1 = torch.nn.LayerNorm(dim)
    self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
                                     qk_scale, attn_drop, proj_drop, dilation,
                                     use_swin=use_swin)
    self.norm2 = torch.nn.LayerNorm(dim)
    self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
    # self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
    self.dropout = torch.nn.Dropout(drop_path)
    self.pos_emb = pos_emb(dim)

  def forward(self, data: torch.Tensor, octree: OctreeT):
    pe = self.pos_emb(data, octree)
    data = pe + data
    attn = self.attention(
        self.norm1(data), octree)
    data = data + self.dropout(attn)
    ffn = self.mlp(self.norm2(data))
    data = data + self.dropout(ffn)
    return data


class OctFormerStage(torch.nn.Module):

  def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
               dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
               qk_scale: Optional[float] = None, attn_drop: float = 0.0,
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
               use_swin: bool = False,
               pos_emb: torch.nn.Module = SinPosEmb,
               use_checkpoint: bool = True, num_blocks: int = 2,
               octformer_block=OctFormerBlock, **kwargs):
    super().__init__()
    self.num_blocks = num_blocks
    self.use_checkpoint = use_checkpoint
    self.interval = interval    # normalization interval
    self.num_norms = (num_blocks - 1) // self.interval

    self.blocks = torch.nn.ModuleList([octformer_block(
        dim=dim, num_heads=num_heads, patch_size=patch_size,
        dilation=1 if (i % 2 == 0) else dilation,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        attn_drop=attn_drop, proj_drop=proj_drop, pos_emb=pos_emb,
        use_swin=((i // 2) % 2 == 1) if use_swin else False,
        drop_path=drop_path[i] if isinstance(
            drop_path, list) else drop_path,
        nempty=nempty, activation=activation) for i in range(num_blocks)])
    # self.norms = torch.nn.ModuleList([
    #         torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

  def forward(self, data: torch.Tensor, octree: OctreeT):
    for i in range(self.num_blocks):
      if self.use_checkpoint and self.training:
        data = checkpoint(self.blocks[i], data, octree, use_reentrant=False)
      else:
        data = self.blocks[i](data, octree)
    return data


class OctFormer(torch.nn.Module):

  def __init__(self,
               channels: int = 192,
               num_blocks: int = 16,
               num_heads: int = 16,
               patch_size: int = 26, dilation: int = 4,
               drop_path: float = 0.5, attn_drop: float = 0.1, proj_drop: float = 0.1,
               pos_emb: torch.nn.Module = SinPosEmb,
               nempty: bool = False, use_checkpoint: bool = True,
               use_swin: bool = False,
               **kwargs):
    super().__init__()
    self.patch_size = patch_size
    self.dilation = dilation
    self.nempty = nempty
    self.use_swin = use_swin

    self.layers = OctFormerStage(
        dim=channels, num_heads=num_heads, patch_size=patch_size,
        # drop_path=torch.linspace(0, drop_path, num_blocks).tolist(),
        dilation=dilation, nempty=nempty, num_blocks=num_blocks,
        attn_drop=attn_drop, proj_drop=proj_drop, pos_emb=pos_emb,
        use_checkpoint=use_checkpoint, use_swin=use_swin)

  def forward(self, data: torch.Tensor, octree: Octree, depth_list: int):
    data_length = data.shape[0]
    if not isinstance(octree, OctreeT):
      octree = OctreeT(octree, data_length, self.patch_size, self.dilation, 
                       self.nempty, depth_list=depth_list, use_swin=self.use_swin)
    data = self.layers(data, octree)
    return data
