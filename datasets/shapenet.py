import os
import ocnn
import torch
import numpy as np

from thsolver import Dataset
from ocnn.octree import Octree, Points


class TransformShape:

  def __init__(self, flags):
    self.flags = flags

    self.volume_sample_num = flags.volume_sample_num
    self.surface_sample_num = flags.surface_sample_num
    self.points_scale = 0.5  # the points are in [-0.5, 0.5]
    self.noise_std = 0.005
    self.tsdf = 0.05         # truncation of SDF

    self.depth = flags.depth
    self.full_depth = flags.full_depth

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def process_points_cloud(self, sample):
    # get the input
    points = torch.from_numpy(sample['points']).float()
    normals = torch.from_numpy(sample['normals']).float()
    points = points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    points_gt = Points(points=points, normals=normals)
    points_gt.clip(min=-1, max=1)
    octree_gt = self.points2octree(points_gt)

    if self.flags.distort:
      # randomly sample points and add noise
      # Since we rescale points to [-1.0, 1.0] in Line 24, we also need to
      # rescale the `noise_std` here to make sure the `noise_std` is always
      # 0.5% of the bounding box size.
      noise_std = torch.rand(1) * self.noise_std / self.points_scale
      points_noise = points + noise_std * torch.randn_like(points)
      normals_noise = normals + noise_std * torch.randn_like(normals)

      # transform noisy points to octree
      points_in = Points(points=points_noise, normals=normals_noise)
      points_in.clip(-1.0, 1.0)
      octree_in = self.points2octree(points_in)
    else:
      points_in = points_gt
      octree_in = octree_gt

    # construct the output dict
    return {'octree_in': octree_in, 'points_in': points_in,
            'octree_gt': octree_gt, 'points_gt': points_gt}

  def sample_volume(self, sample):
    sdf = sample['sdf']
    grad = sample['grad']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.volume_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    sdf = torch.from_numpy(sdf[rand_idx]).float()
    grad = torch.from_numpy(grad[rand_idx]).float()

    # truncate the sdf
    flag = sdf > self.tsdf
    sdf[flag] = self.tsdf
    grad[flag] = 0.0
    flag = sdf < -self.tsdf
    sdf[flag] = -self.tsdf
    grad[flag] = 0.0

    return {'pos': points, 'sdf': sdf, 'grad': grad}

  def sample_surface(self, sample):
    normals = sample['normals']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.surface_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    normals = torch.from_numpy(normals[rand_idx]).float()
    sdf = torch.zeros(self.surface_sample_num)
    return {'pos': points, 'sdf': sdf, 'grad': normals}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    if self.flags.get('load_sdf'):
      samples = self.sample_volume(sample['sdf'])
      surface = self.sample_surface(sample['point_cloud'])
      for key in samples.keys():
        samples[key] = torch.cat([samples[key], surface[key]], dim=0)

      output.update(samples)
    return output


class ReadFile:

  def __init__(self, flags):
    self.flags = flags

  def __call__(self, filename):
    # load the input point cloud
    output = {}
    if self.flags.get('load_pointcloud'):
      filename_pc = os.path.join(filename, 'pointcloud.npz')
      raw = np.load(filename_pc)
      point_cloud = {'points': raw['points'], 'normals': raw['normals']}
      output['point_cloud'] = point_cloud

    # load the target sdfs and gradients
    if self.flags.get('load_sdf'):
      num = self.flags.get('sdf_file_num', 0)
      name = 'sdf_%d.npz' % np.random.randint(num) if num > 0 else 'sdf.npz'
      filename_sdf = os.path.join(filename, name)
      raw = np.load(filename_sdf)
      sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
      output['sdf'] = sdf

    return output


def collate_func(batch):
  output = ocnn.dataset.CollateBatch(merge_points=False)(batch)

  if 'pos' in output:
    bi = [torch.ones(pos.size(0), 1) * i for i, pos in enumerate(output['pos'])]
    batch_idx = torch.cat(bi, dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)

  for key in ['grad', 'sdf', 'occu', 'weight', 'color']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)

  return output


def get_shapenet_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_func
