import os
import ocnn
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import pandas as pd
from thsolver import Dataset
from ocnn.octree import Octree, Points

class TransformShape:

  def __init__(self, flags):
    self.flags = flags

    self.volume_sample_num = flags.volume_sample_num
    self.surface_sample_num = flags.surface_sample_num
    # if self.flags.get('off_surface_sample_num'):
    #   self.off_surface_sample_num = flags.off_surface_sample_num
    # else:
    #   self.off_surface_sample_num = 0
    self.points_scale = flags.points_scale  # the points are in [-0.5, 0.5]
    self.noise_std = 0.005
    self.tsdf = flags.tsdf         # truncation of SDF

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

    # randomly drop some points if max_points is set to avoid OOM
    if self.flags.get('max_points') and points.shape[0] > self.flags.max_points:
      rand_idx = np.random.choice(points.shape[0], size=self.flags.max_points)
      points = points[rand_idx]
      normals = normals[rand_idx]

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

  def sample_off_surface(self, points):
    '''Randomly sample points in the 3D space.'''
    off_surface_sample_num = self.off_surface_sample_num

    # uniformly sampling in the whole 3D sapce
    pos = torch.rand(off_surface_sample_num, 3) * 2 - 1
    # point gradients
    grad = torch.zeros(off_surface_sample_num, 3)
    # norm = torch.sqrt(torch.sum(pos**2, dim=1, keepdim=True)) + 1e-6
    # grad = pos / norm

    # sdf values
    esp = 0.04
    bbmin, bbmax = points.min(dim=0)[0] - esp, points.max(dim=0)[0] + esp
    mask = torch.logical_and(pos > bbmin, pos < bbmax).all(1)  # inbox
    sdf = -1.0 * torch.ones(off_surface_sample_num)
    sdf[mask.logical_not()] = 1.0
    output = {'pos': pos, 'sdf': sdf, 'grad': grad}

    return output

  def rand_drop(self, sample):
    r'''Randomly drop some points to make the dataset more diverse
        and save GPU memory. '''

    if not self.flags.get('rand_drop'):
      return sample  # no rand_drop, return

    # randomly 1 / 8 points
    point_cloud = sample['point_cloud']
    points = point_cloud['points']
    center = np.mean(points, axis=0)
    pc = (points - center) > 0
    idx = (pc * np.array([4, 2, 1])).sum(axis=1)
    rnd = np.random.randint(8)  # random index
    flag = idx == rnd
    point_cloud['points'] = point_cloud['points'][flag]
    point_cloud['normals'] = point_cloud['normals'][flag]

    if self.flags.get('load_sdf'):
      sdf = sample['sdf']
      pc = (sdf['points'] - center) > 0
      idx = (pc * np.array([4, 2, 1])).sum(axis=1)
      flag = idx == rnd        # reuse the same random index
      sdf['points'] = sdf['points'][flag]
      sdf['grad'] = sdf['grad'][flag]
      sdf['sdf'] = sdf['sdf'][flag]
    return {'point_cloud': point_cloud, 'sdf': sdf}

  def __call__(self, sample, idx):
    # sample = self.rand_drop(sample)
    output = self.process_points_cloud(sample['point_cloud'])

    if self.flags.get('load_sdf'):
      samples = self.sample_volume(sample['sdf'])
      surface = self.sample_surface(sample['point_cloud'])
      # off_surface = self.sample_off_surface(surface['pos'])
      for key in samples.keys():
        samples[key] = torch.cat(
            [samples[key], surface[key]], dim=0)

      output.update(samples)

    # Sketch Condition
    if self.flags.get('load_sketch'):
      output['image'] = sample['image']
      output['projection_matrix'] = sample['projection_matrix'].unsqueeze(0)

    if self.flags.get('load_image'):
      output['image'] = sample['image']

    if self.flags.get('load_text'):
      output['text'] = sample['text']

    return output


class ReadFile:

  def __init__(self, flags):
    self.flags = flags
    if flags.get("load_sketch"):
      self.load_sketch = ReadSketch(flags)
    if flags.get("load_image"):
      self.load_image = ReadImage(flags)
    if flags.get("load_text"):
      self.load_text = ReadText(flags)

  def __call__(self, filename):  # , uid=None):
    # load the input point cloud
    output = {}

    uid = '/'.join(filename.split('/')[-2:])
    # print(uid)
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

    # Load the sketch image
    if self.flags.get('load_sketch'):
      img, pm, sketch_view = self.load_sketch(uid)
      output['uid'] = uid
      output['image'] = img
      output['projection_matrix'] = pm
      output['sketch_view'] = sketch_view

    if self.flags.get('load_image'):
      img = self.load_image(uid)
      output['uid'] = uid
      output['image'] = img

    if self.flags.get('load_text'):
      text = self.load_text(uid)
      output['text'] = text

    return output


class ReadImage:
  def __init__(self, flags):
    self.flags = flags
    self.image_folder = flags.image_location

  def load_image(self, uid):
    uid = uid.split('/')[-1]
    img = Image.open(os.path.join(self.image_folder, f'{uid}_0.png')).convert('RGB')
    return img

  def __call__(self, uid):
    return self.load_image(uid)

class ReadText:
  def __init__(self, flags):
    self.flags = flags
    if flags.name == 'shapenet':
      self.read_text2shape()
    elif flags.name == 'objaverse':
      self.read_objaverse()
    else:
      raise ValueError(f'Unsupported dataset: {flags.name}')

  def read_objaverse(self):
    text_csv = pd.read_csv(self.flags.text_location,
                           header=None, names=["uid", "text"])
    self.text_dict = dict(zip(text_csv['uid'], text_csv['text']))

  def read_text2shape(self):
    text_csv = pd.read_csv(self.flags.text_location)
    self.text_dict = text_csv.groupby('modelId')['description'].apply(list).to_dict()
  
  def __call__(self, uid):
    if self.flags.get("text_prompt"):
      return self.flags.text_prompt
    uid = uid.split('/')[-1]
    if uid in self.text_dict:
      texts = self.text_dict[uid]
      if isinstance(texts, str):
        return texts
      if len(texts) == 0:
        return "A 3D model."
      text = random.choice(texts)
      if not isinstance(text, str):
        return "A 3D model."
      return text
    else:
      return "A 3D model."


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
