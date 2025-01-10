# --------------------------------------------------------
# Dual Octree Graph Neural Networks
# Copyright (c) 2024 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np

from thsolver import Dataset
from .shapenet import collate_func
from .shapenet import TransformShape


class TransformScene(TransformShape):

  def __init__(self, flags):
    super().__init__(flags)

    self.point_sample_num = 10000
    self.volume_sample_num = 4096
    self.surface_sample_num = 2048
    self.points_scale = 0.6  # the points are actually in [-0.55, 0.55]
    self.noise_std = 0.005
    self.pos_weight = 10

  def sample_occu(self, sample):
    points, occus = sample['points'], sample['occupancies']
    points = points / self.points_scale
    points = points + 1e-6 * np.random.randn(*points.shape)  # ref ConvoNet
    occus = np.unpackbits(occus)[:points.shape[0]]

    rand_idx = np.random.choice(points.shape[0], size=self.volume_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    occus = torch.from_numpy(occus[rand_idx]).float()
    # 1 - outside shapes; 1 - inside shapes
    occus = 1 - occus  # to be consistent with ShapeNet

    # The number of points inside shapes is roughly 1.2% of points outside
    # shapes, we set weights to 10 for points inside shapes.
    weight = torch.ones_like(occus)
    weight[occus < 0.5] = self.pos_weight

    # The points are not on the surfaces, the gradients are 0
    grad = torch.zeros_like(points)
    return {'pos': points, 'occu': occus, 'weight': weight, 'grad': grad}

  def sample_surface(self, sample):
    points, normals = sample['points'], sample['normals']

    # sample points
    rand_idx = np.random.choice(points.shape[0], size=self.surface_sample_num)
    pos = torch.from_numpy(points[rand_idx])
    grad = torch.from_numpy(normals[rand_idx])
    pos = pos / self.points_scale  # scale to [-1.0, 1.0]
    occus = torch.ones(self.surface_sample_num) * 0.5
    weight = torch.ones(self.surface_sample_num) * 2.0  # TODO: tune this value

    return {'pos': pos, 'occu': occus, 'weight': weight, 'grad': grad}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    # sample ground truth sdfs
    if self.flags.get('load_occu'):
      occus = self.sample_occu(sample['occus'])
      surface_occus = self.sample_surface(sample['point_cloud'])
      for key in occus.keys():
        occus[key] = torch.cat([occus[key], surface_occus[key]], dim=0)
      output.update(occus)

    return output


class ReadFile:

  def __init__(self, load_occu=False):
    self.load_occu = load_occu
    self.num_files = 10

  def __call__(self, filename):
    num = np.random.randint(self.num_files)
    filename_pc = os.path.join(
        filename, 'pointcloud/pointcloud_%02d.npz' % num)
    raw = np.load(filename_pc)
    point_cloud = {'points': raw['points'], 'normals': raw['normals']}
    output = {'point_cloud': point_cloud}

    if self.load_occu:
      num = np.random.randint(self.num_files)
      filename_occu = os.path.join(
          filename, 'points_iou/points_iou_%02d.npz' % num)
      raw = np.load(filename_occu)
      occus = {'points': raw['points'], 'occupancies': raw['occupancies']}
      output['occus'] = occus

    return output


def get_synthetic_room_dataset(flags):
  transform = TransformScene(flags)
  read_file = ReadFile(flags.load_occu)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_func
