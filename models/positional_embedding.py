"""
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import math
from functools import partial
from ocnn.octree import Octree
import ocnn
import numpy as np
FULL_DEPTH = 3
MAX_DEPTH = 6

def init_3d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
  freqs_x = []
  freqs_y = []
  freqs_z = []
  # (dim // 6)
  mag = 1 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
  for i in range(num_heads):
    phi = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
    theta = torch.rand(1) * torch.pi / 2 if rotate else torch.zeros(1)
    # TODO: correct?
    fx = torch.cat([
      mag * torch.sin(theta) * torch.cos(phi), 
      mag * torch.sin(theta) * torch.cos(torch.pi/2 + phi),
      mag * torch.sin(theta + torch.pi/2) * torch.cos(phi),
    ], dim=-1)
    fy = torch.cat([
      mag * torch.sin(theta) * torch.sin(phi), 
      mag * torch.sin(theta) * torch.sin(torch.pi/2 + phi),
      mag * torch.sin(theta + torch.pi/2) * torch.sin(phi),
    ], dim=-1)
    fz = torch.cat([
      mag * torch.cos(theta),
      mag * torch.cos(theta),
      mag * torch.cos(theta + torch.pi/2),
    ], dim=-1)
    freqs_x.append(fx)
    freqs_y.append(fy)
    freqs_z.append(fz)
  freqs_x = torch.stack(freqs_x, dim=0)
  freqs_y = torch.stack(freqs_y, dim=0)
  freqs_z = torch.stack(freqs_z, dim=0)
  freqs = torch.stack([freqs_x, freqs_y, freqs_z], dim=0)
  return freqs


def compute_mixed_cis(freqs: torch.Tensor, xyz: torch.Tensor, num_heads: int):
  N = xyz.shape[0]
  t_x, t_y, t_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
  # No float 16 for this range
  with torch.cuda.amp.autocast(enabled=False):
    freqs_x = (t_x.unsqueeze(-1) @
               freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
    freqs_y = (t_y.unsqueeze(-1) @
               freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
    freqs_z = (t_z.unsqueeze(-1) @
               freqs[2].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
    freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_z)
  return freqs_cis


def compute_axial_cis(dim: int, xyz: torch.Tensor, theta: float = 100.0):
  freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 6, device=xyz.device)
                   [: (dim // 6)].float() / dim))
  freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 6, device=xyz.device)
                   [: (dim // 6)].float() / dim))
  freqs_z = 1.0 / (theta ** (torch.arange(0, dim, 6, device=xyz.device)
                   [: (dim // 6)].float() / dim))

  t_x, t_y, t_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
  # (N, dim // 6)
  freqs_x = torch.outer(t_x, freqs_x)
  freqs_y = torch.outer(t_y, freqs_y)
  freqs_z = torch.outer(t_z, freqs_z)
  # (N, dim // 6)
  freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
  freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
  freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
  return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
    shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
  elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
    shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
  # (num_heads, N, dim // num_heads // 2)
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
  return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RotaryPosEmb(torch.nn.Module):
  """Multi-head Attention block with rotary position embeddings."""

  def __init__(self, dim, num_heads, rope_theta=200.0, rope_mixed=True):
    super().__init__()
    self.rope_mixed = rope_mixed
    self.dim = dim
    self.num_heads = num_heads

    if self.rope_mixed:
      self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

      # (3, )
      freqs = init_3d_freqs(
          dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
          rotate=True
      ).view(3, -1)
      self.freqs = nn.Parameter(freqs, requires_grad=True)
    else:
      self.compute_cis = partial(
          compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)

  def rescale_pos(self, x, scale, max_scale):
    x = x * max_scale // scale
    x += max_scale // scale // 2
    return x

  def forward(self, qkv: torch.Tensor, octree: Octree):
    # Apply rotary position embedding
    N = qkv.shape[0]
    C = self.dim
    H = self.num_heads
    qkv = qkv.view(-1, 3, H, C // H).permute(1, 2, 0, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]

    max_scale = 2 ** (octree.max_depth + 1)
    freqs_cis = []
    for d in range(octree.start_depth, octree.max_depth + 1):
      scale = 2 ** d
      x, y, z, b = octree.xyzb(d)
      x = self.rescale_pos(x, scale, max_scale)
      y = self.rescale_pos(y, scale, max_scale)
      z = self.rescale_pos(z, scale, max_scale)
      xyz = torch.stack([x, y, z], dim=-1).float()

      if self.rope_mixed:
        # (N, dim // num_heads // 2)
        freqs_cis_d = self.compute_cis(self.freqs, xyz)
      else:
        # (N, dim // num_heads // 2)
        freqs_cis_d = self.compute_cis(xyz=xyz)
      freqs_cis.append(freqs_cis_d)
    freqs_cis = torch.cat(freqs_cis, dim=-2)
    q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
    qkv = torch.stack([q, k, v], dim=0)
    qkv = qkv.permute(2, 0, 1, 3).reshape(N, 3 * C)

    return qkv


class SinPosEmb(torch.nn.Module):
  def __init__(self, num_embed: int, full_depth: int = FULL_DEPTH, max_depth: int = MAX_DEPTH):
    super().__init__()
    self.num_embed = num_embed
    self.max_depth = max_depth
    self.full_depth = full_depth
    self.max_scale = 2 ** (self.max_depth + 1)

    self.depth_emb = torch.nn.Embedding(
        self.max_depth - self.full_depth + 1, num_embed)

  def rescale_pos(self, x, scale):
    x = x * self.max_scale // scale
    x += self.max_scale // scale // 2
    return x

  def get_emb(self, sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

  def get_3d_pos_emb(self, pos_x, pos_y, pos_z):
    device = pos_x.device

    channels = int(np.ceil(self.num_embed / 6) * 2)
    if channels % 2:
      channels += 1

    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels,
                      2, device=device).float() / channels))

    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = self.get_emb(sin_inp_x)
    emb_y = self.get_emb(sin_inp_y)
    emb_z = self.get_emb(sin_inp_z)
    emb = torch.zeros(
        (pos_x.shape[0], channels * 3),
        device=device,
    )
    emb[:, : channels] = emb_x
    emb[:, channels: 2 * channels] = emb_y
    emb[:, 2 * channels:] = emb_z

    return emb[:, :self.num_embed]

  def forward(self, data: torch.Tensor, octree: Octree):
    position_embeddings = []
    for d in range(octree.start_depth, octree.max_depth + 1):
      scale = 2 ** d
      x, y, z, b = octree.xyzb(d)
      x = self.rescale_pos(x, scale)
      y = self.rescale_pos(y, scale)
      z = self.rescale_pos(z, scale)

      pos_emb_d = self.get_3d_pos_emb(x, y, z)
      depth_emb_d = self.depth_emb(torch.tensor(
          [d - self.full_depth], device=octree.device))
      position_embeddings.append(pos_emb_d + depth_emb_d)
    position_embeddings = torch.cat(position_embeddings, dim=0)
    return position_embeddings


class OctreeConvPosEmb(torch.nn.Module):
  def __init__(self, num_embed: int, full_depth: int = FULL_DEPTH, max_depth: int = MAX_DEPTH, groups: int = 32, nempty: bool = False):
    super().__init__()
    self.full_depth = full_depth
    self.max_depth = max_depth
    self.conv = torch.nn.ModuleList([
        ocnn.modules.OctreeConvGnRelu(
            in_channels=num_embed, out_channels=num_embed, group=groups, nempty=nempty)
        for i in range(max_depth - full_depth + 1)])
    self.depth_emb = torch.nn.Embedding(
        self.max_depth - self.full_depth + 1, num_embed)

  def forward(self, data: torch.Tensor, octree: Octree):
    position_embeddings = []
    cur_seq_len = 0
    for d in range(octree.start_depth, octree.max_depth + 1):
      nnum_d = octree.nnum[d]
      # clone the data to avoid in-place operation
      data_d = data[cur_seq_len:cur_seq_len + nnum_d].clone()
      pos_emb_d = self.conv[d - octree.start_depth](data_d, octree, d)
      depth_emb_d = self.depth_emb(torch.tensor(
          [d - self.full_depth], device=octree.device))
      position_embeddings.append(pos_emb_d + depth_emb_d)
      cur_seq_len += nnum_d
      if cur_seq_len >= data.shape[0]:
        break
    position_embeddings = torch.cat(position_embeddings, dim=0)
    return position_embeddings

class DepthPosEmb(torch.nn.Module):
  def __init__(self, num_embed: int, full_depth: int = FULL_DEPTH, max_depth: int = MAX_DEPTH):
    super().__init__()
    self.full_depth = full_depth
    self.max_depth = max_depth
    self.depth_emb = torch.nn.Embedding(
        self.max_depth - self.full_depth + 1, num_embed)

  def forward(self, data: torch.Tensor, octree: Octree):
    depth_embedding = []
    for d in range(octree.start_depth, octree.max_depth + 1):
      nnum_d = octree.nnum[d]
      # clone the data to avoid in-place operation
      depth_emb_d = self.depth_emb(torch.tensor(
          [d - self.full_depth], device=octree.device)).repeat(nnum_d, 1)
      depth_embedding.append(depth_emb_d)
    depth_embedding = torch.cat(depth_embedding, dim=0)
    return depth_embedding