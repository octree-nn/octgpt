# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import numpy as np
import torch
import ocnn
import dwconv

from ocnn.octree import Octree
from typing import Optional, List
from torch.utils.checkpoint import checkpoint
from positional_encodings.torch_encodings import PositionalEncoding3D

class OctreeT(Octree):

    def __init__(self, octree: Octree, data_length: int, patch_size: int = 24, dilation: int = 4,
                nempty: bool = True, max_depth: Optional[int] = None,
                start_depth: Optional[int] = None, **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.patch_size = patch_size
        self.dilation = dilation    # TODO dilation as a list
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1

        self.block_num = patch_size * dilation

        self.nnum_t = data_length
        self.nnum_a = int(np.ceil(self.nnum_t / self.block_num) * self.block_num)
        self.batch_idx = None
        self.patch_mask = None
        self.dilate_mask = None
        self.rel_pos = None
        self.dilate_pos = None
        self.build_t()

    def build_t(self):
        self.build_batch_idx()
        self.build_attn_mask()
        self.build_rel_pos()

    def build_batch_idx(self):
        batch = []
        for d in range(self.start_depth, self.max_depth + 1):
            batch.append(self.batch_id(d, self.nempty))
        self.batch_idx = torch.cat(batch, dim=0)
        self.batch_idx = self.patch_partition(self.batch_idx[:self.nnum_t], self.batch_size)

    def build_attn_mask(self):
        batch = self.batch_idx
        mask = batch.view(-1, self.patch_size)
        self.patch_mask = self._calc_attn_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilate_mask = self._calc_attn_mask(mask)

    def _calc_attn_mask(self, mask: torch.Tensor):
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def build_rel_pos(self):
        xyz = []
        for d in range(self.start_depth, self.max_depth + 1):
            key = self.key(d, self.nempty)
            x, y, z, _ = ocnn.octree.key2xyz(key, d)
            xyz.append(torch.stack([x, y, z], dim=1))
        xyz = torch.cat(xyz, dim=0)
        xyz = self.patch_partition(xyz[:self.nnum_t])

        xyz = xyz.view(-1, self.patch_size, 3)
        self.rel_pos = xyz.unsqueeze(2) - xyz.unsqueeze(1)

        xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
        xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
        self.dilate_pos = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    def patch_partition(self, data: torch.Tensor, fill_value=0):
        assert data.shape[0] == self.nnum_t
        num = self.nnum_a - self.nnum_t
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    def patch_reverse(self, data: torch.Tensor):
        return data[:self.nnum_t]


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


class OctreeDWConvBn(torch.nn.Module):

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                stride: int = 1, nempty: bool = False):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
                in_channels, kernel_size, nempty, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out


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

class SinPosEmb(torch.nn.Module):
    def __init__(self, n_embed, num_depth=3):
        super().__init__()
        self.n_embed = n_embed
        self.pos_emb = PositionalEncoding3D(n_embed)
        self.depth_emb = torch.nn.Embedding(3, n_embed)
    
    def forward(self, octree, depth_low, depth_high):
        position_embeddings = []
        for d in range(depth_low, depth_high + 1):
            scale = 2 ** d
            x, y, z, b = octree.xyzb(d)
            pos_emb_d = self.pos_emb(torch.zeros((1, scale, scale, scale, self.n_embed), device=octree.device))
            pos_emb_d = pos_emb_d[0, x, y, z, :]
            depth_emb_d = self.depth_emb(torch.tensor([d - depth_low], device=octree.device))
            position_embeddings.append(pos_emb_d + depth_emb_d)
        position_embeddings = torch.cat(position_embeddings, dim=0)
        return position_embeddings


class OctreeAttention(torch.nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
            qkv_bias: bool = True, qk_scale: Optional[float] = None,
            attn_drop: float = 0.0, proj_drop: float = 0.0,
            dilation: int = 1, use_rpe: bool = False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
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

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation

        # patch partition
        data = octree.patch_partition(data)
        if D > 1:    # dilation
            rel_pos = octree.dilate_pos
            mask = octree.dilate_mask
            data = data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        else:
            rel_pos = octree.rel_pos
            mask = octree.patch_mask
        
        # teacher forcing mask
        teacher_forcing_mask = torch.tril(torch.ones(K, K, device=data.device)).unsqueeze(0)
        mask = mask.masked_fill(teacher_forcing_mask == 0, -1e3)

        data = data.view(-1, K, C)

        # qkv
        qkv = self.qkv(data).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (N, H, K, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)                # (N, H, K, K)
        attn = self.apply_rpe(attn, rel_pos)    # (N, H, K, K)
        attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        data = (attn @ v).transpose(1, 2).reshape(-1, C)

        # patch reverse
        if D > 1:    # dilation
            data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        data = octree.patch_reverse(data)

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
            activation: torch.nn.Module = torch.nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
            qk_scale, attn_drop, proj_drop, dilation)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        # self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.dropout = torch.nn.Dropout(drop_path)
        # self.cpe = OctreeDWConvBn(dim, nempty=nempty)
        self.pe = SinPosEmb(dim)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth_low: int, depth_high: int):
        data = self.pe(octree, depth_low, depth_high)[:octree.nnum_t] + data
        attn = self.attention(self.norm1(data), octree, depth_low)
        # data = data + self.drop_path(attn, octree, depth_low)
        data = data + self.dropout(attn)
        ffn = self.mlp(self.norm2(data))
        # data = data + self.drop_path(ffn, octree, depth_low)
        data = data + self.dropout(ffn)
        return data


class OctFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
            dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
            qk_scale: Optional[float] = None, attn_drop: float = 0.0,
            proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
            activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
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
            attn_drop=attn_drop, proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            nempty=nempty, activation=activation) for i in range(num_blocks)])
        # self.norms = torch.nn.ModuleList([
        #         torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

    def forward(self, data: torch.Tensor, octree: OctreeT, depth_low: int, depth_high: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(self.blocks[i], data, octree, depth_low, depth_high, use_reentrant=False)
            else:
                data = self.blocks[i](data, octree, depth_low, depth_high)
            # if i % self.interval == 0 and i != 0:
            #     data = self.norms[(i - 1) // self.interval](data)
        return data


class PatchEmbed(torch.nn.Module):

    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
            nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty)
            for i in range(self.num_stages)])
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
            kernel_size: List[int] = [2], nempty: bool = True):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
            stride=2, nempty=nempty, use_bias=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class OctFormer(torch.nn.Module):

    def __init__(self, 
            channels: int = 192,
            num_blocks: int = 16,
            num_heads: int = 16,
            patch_size: int = 26, dilation: int = 4, 
            drop_path: float = 0.5, attn_drop: float = 0.1, proj_drop: float = 0.1,
            nempty: bool = False, use_checkpoint: bool = True,
            **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        
        # self.patch_embed = ocnn.modules.OctreeConvBnRelu(in_channels, channels, kernel_size=[3], stride=1, nempty=nempty)
        self.layers = OctFormerStage(
            dim=channels, num_heads=num_heads, patch_size=patch_size,
            # drop_path=torch.linspace(0, drop_path, num_blocks).tolist(),
            dilation=dilation, nempty=nempty, num_blocks=num_blocks,
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_checkpoint=use_checkpoint)
            

    def forward(self, data: torch.Tensor, octree: Octree, depth_low: int, depth_high: int):
        # data = self.patch_embed(data, octree, depth)
        # from .octformer_backup import OctreeT
        octree = OctreeT(octree, data.shape[0], self.patch_size, self.dilation, self.nempty, max_depth=depth_high, start_depth=depth_low)
        data = self.layers(data, octree, depth_low, depth_high)
        return data
