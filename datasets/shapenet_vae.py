
import os
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
    colors = torch.from_numpy(sample['colors']).float()
    points = points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    features = colors if self.flags.load_color else None
    points_gt = Points(points=points, normals=normals, features=features)
    points_gt.clip(min=-1, max=1)
    octree_gt = self.points2octree(points_gt)

    # construct the output dict
    return {'octree_in': octree_gt, 'octree_gt': octree_gt, 'points_in': points}

  def sample_volume(self, sample):
    sdf = sample['sdf']
    grad = sample['grad']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.volume_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    sdf = torch.from_numpy(sdf[rand_idx]).float()
    grad = torch.from_numpy(grad[rand_idx]).float()
    color = torch.zeros(self.volume_sample_num, 3)
    return {'pos': points, 'sdf': sdf, 'grad': grad, 'color': color}

  def sample_surface(self, sample):
    normals = sample['normals']
    colors = sample['colors']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.surface_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    colors = torch.from_numpy(colors[rand_idx]).float()
    normals = torch.from_numpy(normals[rand_idx]).float()
    sdf = torch.zeros(self.surface_sample_num)
    return {'pos': points, 'sdf': sdf, 'grad': normals, 'color': colors}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    samples = self.sample_volume(sample['sdf'])
    surface = self.sample_surface(sample['point_cloud'])
    for key in samples.keys():
      samples[key] = torch.cat([samples[key], surface[key]], dim=0)
    if not self.flags.load_color:
      samples.pop('color', None)

    output.update(samples)
    return output


class ReadFile:

  def __init__(self, load_color=False):
    self.load_color = load_color

  def __call__(self, filename):
    filename_pc = os.path.join(filename, 'pointcloud.npz')
    raw = np.load(filename_pc)
    point_cloud = {'points': raw['points'], 'normals': raw['normals']}

    filename_sdf = os.path.join(filename, 'sdf.npz')
    raw = np.load(filename_sdf)
    sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}

    if self.load_color:
      filename_color = os.path.join(filename, 'color.npz')
      raw = np.load(filename_color)
      point_cloud['colors'] = raw['colors']  # colors of the input point clouds
    else:
      point_cloud['colors'] = np.zeros_like(point_cloud['normals'])

    return {'point_cloud': point_cloud, 'sdf': sdf}


def collate_func(batch):
  output = ocnn.dataset.CollateBatch(merge_points=False)(batch)

  if 'pos' in output:
    batch_idx = torch.cat([torch.ones(pos.size(0), 1) * i
                           for i, pos in enumerate(output['pos'])], dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)

  for key in ['grad', 'sdf', 'occu', 'weight', 'color']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)

  return output



def get_shapenet_vae_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags.load_color)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
