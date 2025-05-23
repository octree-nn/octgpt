# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

# autopep8: off
import os
import torch
import torch.autograd
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import trimesh
import copy

from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from ocnn.octree import Octree, Points
from ocnn.nn import octree2voxel, octree_pad, octree_depad
from ocnn.utils import meshgrid
from tqdm import tqdm
# autopep8: on


def get_mgrid(size, dim=3):
  r''' Example:
  >>> get_mgrid(3, dim=2)
          array([[0.0,    0.0],
          [0.0,    1.0],
          [0.0,    2.0],
          [1.0,    0.0],
          [1.0,    1.0],
          [1.0,    2.0],
          [2.0,    0.0],
          [2.0,    1.0],
          [2.0,    2.0]], dtype=float32)
  '''
  coord = np.arange(0, size, dtype=np.float32)
  coords = [coord] * dim
  output = np.meshgrid(*coords, indexing='ij')
  output = np.stack(output, -1)
  output = output.reshape(size**dim, dim)
  return output


def lin2img(tensor):
  channels = 1
  num_samples = tensor.shape
  size = int(np.sqrt(num_samples))
  return tensor.view(channels, size, size)


def make_contour_plot(array_2d, mode='log'):
  fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

  if (mode == 'log'):
    nlevels = 6
    levels_pos = np.logspace(-2, 0, num=nlevels)    # logspace
    levels_neg = -1. * levels_pos[::-1]
    levels = np.concatenate(
        (levels_neg, np.zeros((0)), levels_pos), axis=0)
    colors = plt.get_cmap("Spectral")(
        np.linspace(0., 1., num=nlevels * 2 + 1))
  elif (mode == 'lin'):
    nlevels = 10
    levels = np.linspace(-.5, .5, num=nlevels)
    colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=nlevels))
  else:
    raise NotImplementedError

  sample = np.flipud(array_2d)
  CS = ax.contourf(sample, levels=levels, colors=colors)
  fig.colorbar(CS)

  ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
  ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
  ax.axis('off')
  return fig


def write_sdf_summary(model, writer, global_step, alias=''):
  size = 128
  coords_2d = get_mgrid(size, dim=2)
  coords_2d = coords_2d / size - 1.0     # [0, size] -> [-1, 1]
  coords_2d = torch.from_numpy(coords_2d)
  with torch.no_grad():
    zeros = torch.zeros_like(coords_2d[:, :1])
    ones = torch.ones_like(coords_2d[:, :1])
    names = ['train_yz_sdf_slice',
             'train_xz_sdf_slice', 'train_xy_sdf_slice']
    coords = [torch.cat((zeros, coords_2d), dim=-1),
              torch.cat((coords_2d[:, :1], zeros,
                        coords_2d[:, -1:]), dim=-1),
              torch.cat((coords_2d, -0.75 * ones), dim=-1)]
    for name, coord in zip(names, coords):
      ids = torch.zeros(coord.shape[0], 1)
      coord = torch.cat([coord, ids], dim=1).cuda()
      sdf_values = model(coord)
      sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
      fig = make_contour_plot(sdf_values)
      writer.add_figure(alias + name, fig, global_step=global_step)


def calc_field_values(model, size: int = 256, max_batch: int = 64**3,
                      bbmin: float = -1.0, bbmax: float = 1.0, channel: int = 1):
  # generate samples
  num_samples = size ** 3
  samples = get_mgrid(size, dim=3)
  samples = samples * ((bbmax - bbmin) / size) + \
      bbmin    # [0,sz]->[bbmin,bbmax]
  samples = torch.from_numpy(samples)
  out = torch.zeros(num_samples, channel)

  # forward
  head = 0
  while head < num_samples:
    tail = min(head + max_batch, num_samples)
    sample_subset = samples[head:tail, :]
    idx = torch.zeros(sample_subset.shape[0], 1)
    pts = torch.cat([sample_subset, idx], dim=1).cuda()
    pred = model(pts).view(-1, channel).detach().cpu()
    out[head:tail] = pred
    head += max_batch
  out = out.reshape(size, size, size, channel).squeeze().numpy()
  return out


def marching_cubes(values, level=0, with_color=False):
  colors = None
  vtx = np.zeros((0, 3))
  faces = np.zeros((0, 3))

  try:
    if not with_color:
      vtx, faces, _, _ = skimage.measure.marching_cubes(values, level)
    else:
      import marching_cubes as mcubes
      sdfs, colors = values[..., 0], values[..., 1:].clip(0, 1)
      vtx_with_color, faces = mcubes.marching_cubes_color(
          sdfs, colors, level)
      vtx, colors = vtx_with_color[:, :3], vtx_with_color[:, 3:]
  except:
    pass

  return vtx, faces, colors


def create_mesh(model, filename, size=256, max_batch=64**3, level=0,
                bbmin=-0.9, bbmax=0.9, mesh_scale=1.0, save_sdf=False,
                with_color=False, clean=False, **kwargs):
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  channel = 1 if not with_color else 4
  values = calc_field_values(model, size, max_batch, bbmin, bbmax, channel)
  vtx, faces, colors = marching_cubes(values, level, with_color)

  # normalize vtx
  vtx = vtx * ((bbmax - bbmin) / size) + bbmin     # [0,sz]->[bbmin,bbmax]
  vtx = vtx * mesh_scale                                                 # rescale

  # save to ply and npy
  mesh = trimesh.Trimesh(vtx, faces, vertex_colors=colors)
  if clean:
    components = mesh.split(only_watertight=True)
    # bbox = []
    # for c in components:
    #   bbmin = c.vertices.min(0)
    #   bbmax = c.vertices.max(0)
    #   bbox.append((bbmax - bbmin).max())
    # max_component = np.argmax(bbox)
    mesh = trimesh.util.concatenate(components)
  mesh.export(filename)
  if save_sdf:
    np.save(filename[:-4] + ".sdf.npy", values)


def calc_sdf_err(filename_gt, filename_pred):
  scale = 1.0e2    # scale the result for better display
  sdf_gt = np.load(filename_gt)
  sdf = np.load(filename_pred)
  err = np.abs(sdf - sdf_gt).mean() * scale
  return err


def calc_chamfer(filename_gt, filename_pred, point_num):
  scale = 1.0e5    # scale the result for better display
  np.random.seed(101)

  mesh_a = trimesh.load(filename_gt)
  points_a, _ = trimesh.sample.sample_surface(mesh_a, point_num)
  mesh_b = trimesh.load(filename_pred)
  points_b, _ = trimesh.sample.sample_surface(mesh_b, point_num)

  kdtree_a = cKDTree(points_a)
  dist_a, _ = kdtree_a.query(points_b)
  chamfer_a = np.mean(np.square(dist_a)) * scale

  kdtree_b = cKDTree(points_b)
  dist_b, _ = kdtree_b.query(points_a)
  chamfer_b = np.mean(np.square(dist_b)) * scale
  return chamfer_a, chamfer_b


def points2ply(filename: str, points: Points):
  # data types
  data = points.points.numpy()
  py_types = (float, float, float)
  npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
  if points.normals is not None:
    normal = points.normals.numpy()
    py_types = py_types + (float, float, float)
    npy_types = npy_types + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    data = np.concatenate((data, normal), axis=1)

  # format into NumPy structured array
  vertices = []
  for idx in range(data.shape[0]):
    vertices.append(tuple(dtype(d) for dtype, d in zip(py_types, data[idx])))
  structured_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(structured_array, 'vertex')

  # write ply
  PlyData([el]).write(filename)


def data2voxel(data, octree, full_depth):
  batch_size = octree.batch_size
  voxel_size = 2 ** full_depth
  data_full = octree2voxel(data=data, octree=octree, depth=full_depth)
  data_full = data_full.reshape(batch_size, voxel_size ** 3, -1)
  return data_full


def voxel2data(data_full, octree, full_depth):
  batch_size = octree.batch_size
  voxel_size = 2 ** full_depth
  x, y, z, b = octree.xyzb(full_depth)
  data = data_full.reshape(batch_size, voxel_size,
                           voxel_size, voxel_size, -1)[b, x, y, z]
  return data


def voxel2mesh(voxel, threshold=0.4, use_vertex_normal: bool = False):
  verts, faces, vertex_normals = _voxel2mesh(voxel, threshold)
  if use_vertex_normal:
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
  else:
    return trimesh.Trimesh(vertices=verts, faces=faces)


def _voxel2mesh(voxels, threshold=0.5):

  top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
  top_faces = [[0, 1, 3], [1, 2, 3]]
  top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

  bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
  bottom_faces = [[1, 0, 3], [2, 1, 3]]
  bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

  left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
  left_faces = [[0, 1, 3], [2, 0, 3]]
  left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

  right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
  right_faces = [[1, 0, 3], [0, 2, 3]]
  right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

  front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
  front_faces = [[1, 0, 3], [0, 2, 3]]
  front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

  back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
  back_faces = [[0, 1, 3], [2, 0, 3]]
  back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

  vert_scale = 1.0
  top_verts = np.array(top_verts) * vert_scale
  top_faces = np.array(top_faces)
  bottom_verts = np.array(bottom_verts) * vert_scale
  bottom_faces = np.array(bottom_faces)
  left_verts = np.array(left_verts) * vert_scale
  left_faces = np.array(left_faces)
  right_verts = np.array(right_verts) * vert_scale
  right_faces = np.array(right_faces)
  front_verts = np.array(front_verts) * vert_scale
  front_faces = np.array(front_faces)
  back_verts = np.array(back_verts) * vert_scale
  back_faces = np.array(back_faces)

  dim = voxels.shape[0]
  new_voxels = np.zeros((dim+2, dim+2, dim+2))
  new_voxels[1:dim+1, 1:dim+1, 1:dim+1] = voxels
  voxels = new_voxels

  scale = 2/dim
  verts = []
  faces = []
  vertex_normals = []
  curr_vert = 0
  a, b, c = np.where(voxels > threshold)

  for i, j, k in zip(a, b, c):
    if voxels[i, j, k+1] < threshold:
      verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
      faces.extend(top_faces + curr_vert)
      vertex_normals.extend(top_normals)
      curr_vert += len(top_verts)

    if voxels[i, j, k-1] < threshold:
      verts.extend(
          scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
      faces.extend(bottom_faces + curr_vert)
      vertex_normals.extend(bottom_normals)
      curr_vert += len(bottom_verts)

    if voxels[i-1, j, k] < threshold:
      verts.extend(scale * (left_verts +
                   np.array([[i-1, j-1, k-1]])))
      faces.extend(left_faces + curr_vert)
      vertex_normals.extend(left_normals)
      curr_vert += len(left_verts)

    if voxels[i+1, j, k] < threshold:
      verts.extend(scale * (right_verts +
                   np.array([[i-1, j-1, k-1]])))
      faces.extend(right_faces + curr_vert)
      vertex_normals.extend(right_normals)
      curr_vert += len(right_verts)

    if voxels[i, j+1, k] < threshold:
      verts.extend(scale * (front_verts +
                   np.array([[i-1, j-1, k-1]])))
      faces.extend(front_faces + curr_vert)
      vertex_normals.extend(front_normals)
      curr_vert += len(front_verts)

    if voxels[i, j-1, k] < threshold:
      verts.extend(scale * (back_verts +
                   np.array([[i-1, j-1, k-1]])))
      faces.extend(back_faces + curr_vert)
      vertex_normals.extend(back_normals)
      curr_vert += len(back_verts)

  return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)


def export_octree(octree, depth, save_dir=None, index=0):
  try:
    os.makedirs(save_dir, exist_ok=True)
  except FileExistsError:
    pass

  batch_id = octree.batch_id(depth=depth, nempty=False)
  data = torch.ones((len(batch_id), 1), device=octree.device)
  data = octree2voxel(data=data, octree=octree, depth=depth, nempty=False)
  data = data.permute(0, 4, 1, 2, 3).contiguous()

  batch_size = octree.batch_size

  for i in tqdm(range(batch_size)):
    voxel = data[i].squeeze().cpu().numpy()
    mesh = voxel2mesh(voxel)
    if batch_size == 1:
      mesh.export(os.path.join(save_dir, f'{index}.obj'))
    else:
      mesh.export(os.path.join(save_dir, f'{index + i}.obj'))


def octree2split(octree, depth_low, depth_high, shift=False):
  child = octree.children[depth_high - 1]
  split = (child >= 0).unsqueeze(-1)

  for d in range(depth_low, depth_high - 1)[::-1]:
    split_dim = (2 ** (3 * (depth_high - d - 1)))
    split = split.reshape(-1, split_dim)
    split = octree_pad(data=split, octree=octree, depth=d)

  split = split.float()
  if shift:
    split = 2 * split - 1    # scale to [-1, 1]

  return split


def split2octree(octree, split, depth_low, depth_high, threshold=0.0):

  discrete_split = (split > threshold).float()

  octree_out = copy.deepcopy(octree)
  for d in range(depth_low, depth_high):
    split_i = copy.deepcopy(discrete_split)
    split_dim = 2 ** (3 * (depth_high - d - 1))
    split_i = split_i.reshape(-1, split_dim)
    split_i_sum = torch.sum(split_i, dim=1)
    label = (split_i_sum > 0).long()
    discrete_split = split_i[label.bool()]
    if torch.numel(label) == 0:
      label = torch.zeros((8,), dtype=torch.long, device=octree.device)
    octree_out.octree_split(label, depth=d)
    octree_out.octree_grow(d + 1)
  return octree_out


def octree2seq(octree: Octree, depth_low: int, depth_high: int,
               shift: bool = False):
  seq = torch.cat(octree.children[depth_low:depth_high])
  seq = (seq >= 0).long()

  if shift:  # scale to [-1, 1]
    seq = 2 * seq - 1
  return seq


def seq2octree(octree, seq, depth_low, depth_high, threshold=0.0):
  discrete_seq = (seq > threshold).long()

  octree_out = copy.deepcopy(octree)
  cur_nnum = 0
  for d in range(depth_low, depth_high):
    nnum_d = octree_out.nnum[d]
    label = copy.deepcopy(discrete_seq[cur_nnum:cur_nnum + nnum_d])
    cur_nnum += nnum_d
    if torch.numel(label) == 0:
      label = torch.zeros((8,), dtype=torch.long, device=octree.device)
    octree_out.octree_split(label, depth=d)
    octree_out.octree_grow(d + 1)
  return octree_out


def get_batch_id(octree, depth_list):
  batch_id = []
  for d in depth_list:
    batch_id.append(octree.batch_id(d))
  batch_id = torch.cat(batch_id, dim=0)
  return batch_id


def get_depth2batch_indices(octree, depth_list, buffer_size=None, mask=None):
  '''
  Rearange data from depth-by-depth to batch-by-batch
  # From: [depth3: batch1, batch2, ...,
           depth4: batch1, batch2, ...,
           depth5: batch1, batch2, ...]

  # To:   [batch1: depth3, depth4, ...
           batch2: depth3, depth4, ...
           batch3: depth3, depth4, ...]
  '''
  batch_id = get_batch_id(octree, depth_list)
  if buffer_size is not None:
    batch_buffer = torch.arange(
        octree.batch_size, device=octree.device).reshape(-1, 1)
    batch_buffer = batch_buffer.repeat(1, buffer_size).reshape(-1)
    batch_id = torch.cat([batch_buffer, batch_id])
  if mask is not None:
    batch_id = batch_id[~mask]
  batch_id_sorted, indices = torch.sort(batch_id)
  return batch_id_sorted, indices


def depth2batch(data, indices):
  seq = data[indices]
  return seq


def batch2depth(seq, indices):
  data = torch.zeros_like(seq)
  data = data.scatter_(0, indices.unsqueeze(1).expand_as(seq), seq)
  return data


def zorder2raster(zorder, octree, depth_list):
  xyz = []
  for d in depth_list:
    x, y, z, b = octree.xyzb(d)
    xyz.append(d * 262144 + x * 4096 + y * 64 + z)
  xyz = torch.cat(xyz)
  sorted_xyz, indices = torch.sort(xyz, dim=0, descending=False)
  raster = zorder[indices]
  return raster, indices


def raster2zorder(raster, indices):
  zorder = torch.zeros_like(raster)
  zorder = zorder.scatter_(0, indices.unsqueeze(1).expand_as(raster), raster)
  return zorder


def set_requires_grad(model, bool):
  for p in model.parameters():
    p.requires_grad = bool


def sample(logits, top_k=None, top_p=None, temperature=1.0):
  logits = logits / temperature
  probs = F.softmax(logits, dim=-1)

  # top-k
  if top_k is not None:
    topk, indices = torch.topk(probs, k=top_k, dim=-1)
    probs = torch.zeros(
        *probs.shape).to(probs.device).scatter_(1, indices, topk)

  # top-p
  if top_p is not None:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0

  ix = torch.multinomial(probs, num_samples=1)
  return ix.squeeze(1)


def octree_copy_unpool(data: torch.Tensor, octree: Octree, depth: int, nempty: bool = False):
  r''' Performs octree max unpooling.

  Args:
    data (torch.Tensor): The input tensor.
    octree (Octree): The corresponding octree.
    depth (int): The depth of current data. After unpooling, the corresponding
        depth increases by 1.
  '''

  if not nempty:
    data = octree_depad(data, octree, depth)
  num, channel = data.shape
  out = torch.zeros(num, 8, channel, dtype=data.dtype, device=data.device)
  i = torch.arange(num, dtype=torch.int, device=data.device)
  k = torch.arange(channel, dtype=torch.int, device=data.device)
  i, k = meshgrid(i, k, indexing='ij')
  out[i, :, k] = data.unsqueeze(-1)
  out = out.view(-1, channel)
  if nempty:
    out = octree_depad(out, octree, depth+1)
  return out


def get_filenames(filelist):
  with open(filelist, 'r') as fid:
    lines = fid.readlines()
  filenames = [line.split()[0] for line in lines]
  return filenames


def scale_to_unit_cube(mesh):
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()

  vertices = mesh.vertices - mesh.bounding_box.centroid
  vertices *= 2 / np.max(mesh.bounding_box.extents)

  return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


class TorchRecorder:
  def __init__(self):
    self.total_time = 0
    self.calls = 0
    self.total_memory = 0

  def __enter__(self):
    self.start = torch.cuda.Event(enable_timing=True)
    self.end = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    self.start.record()

  def __exit__(self, *args):
    self.end.record()
    torch.cuda.synchronize()
    self.total_time += self.start.elapsed_time(self.end)
    self.calls += 1
    peak_memory = torch.cuda.memory.max_memory_allocated() / (2 ** 30)
    self.total_memory += peak_memory

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.total_memory = 0

  def avg_time(self):
    return self.total_time / self.calls if self.calls > 0 else 0

  def avg_memory(self):
    return self.total_memory / self.calls if self.calls > 0 else 0
