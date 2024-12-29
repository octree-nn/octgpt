import os
import ocnn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from thsolver import Dataset
from ocnn.octree import Octree, Points
from .sketch_utils import Projection_List, Projection_List_zero

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
      for key in samples.keys():
        samples[key] = torch.cat([samples[key], surface[key]], dim=0)

      output.update(samples)
    
    ## Sketch Condition
    if self.flags.get('load_sketch'):
      output['image'] = sample['image'].unsqueeze(0)
      output['projection_matrix'] = sample['projection_matrix'].unsqueeze(0)

    if self.flags.get('load_image'):
      output['image'] = sample['image'].unsqueeze(0)  
    
    return output


class ReadFile:

  def __init__(self, flags):
    self.flags = flags

  def __call__(self, filename): #, uid=None):
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
      if uid is None:
        raise ValueError('uid should be provided when loading image')
      read_image = ReadSketch(self.flags)
      img, pm, sketch_view = read_image(uid)
      output['uid'] = uid
      output['image'] = img
      output['projection_matrix'] = pm
      output['sketch_view'] = sketch_view
    
    if self.flags.get('load_image'):
      if uid is None:
        raise ValueError('uid should be provided when loading image')
      read_image = ReadRenderedImage(self.flags)
      img = read_image(uid)
      output['uid'] = uid
      output['image'] = img

    return output


class ReadRenderedImage:
  def __init__(self, flags):
    self.flags = flags
    self.image_folder = flags.image_location
    self.to_tensor = transforms.ToTensor()
    self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    self.resize = transforms.Resize((256, 256), antialias=False)
  
  def process_image(self, img):
    img_t = self.to_tensor(img)
    assert img_t.shape[0] == 4
    _, oh, ow = img_t.shape
    ls = max(oh, ow)
    pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
    pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
    
    img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)[0]
    
    img_t[:3] = self.normalize(img_t[:3])
    img_t = self.resize(img_t)
    return img_t
  
  def load_image(self, uid):
    uid = uid.split('/')[-1]
    img = Image.open(os.path.join(self.image_folder, f'{uid}_0.png')).convert('RGBA')
    return self.process_image(img)
  
  def __call__(self, uid):
    return self.load_image(uid)


SKETCH_PER_VIEW = 10
class ReadSketch:
  def __init__(self, flags, 
               feature_drop_out: float = 0.1,
               elevation_zero: bool = False):
    self.flags = flags
    self.feature_folder = os.path.join(flags.image_location, 'edge')
    self.matrix_foler = os.path.join(flags.image_location, 'angles')
    self.feature_drop_out = feature_drop_out
    self.elevation_zero = elevation_zero 
    if self.elevation_zero:
      self.projection_list = Projection_List_zero
    else:
      self.projection_list = Projection_List
    self.to_tensor = transforms.ToTensor()
    self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    self.resize = transforms.Resize((256, 256), antialias=False)
  
  def process_image(self, img):
    img_t = self.to_tensor(img)
    assert img_t.shape[0] == 4
    _, oh, ow = img_t.shape
    ls = max(oh, ow)
    pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
    pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
    
    img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)[0]
    
    img_t[:3] = self.normalize(img_t[:3])
    img_t = self.resize(img_t)
    return img_t
  
  def random_load_image(self, uid):
    if 'input_image' in self.flags:
      img = Image.open(self.flags.input_image).convert('RGBA')
      sketch_view_index = int(self.flags.input_image.split("_")[-2])
    else:
      sketch_view_index = np.random.randint(0, 5 * SKETCH_PER_VIEW)
      img = Image.open(os.path.join(
        self.feature_folder, 
        uid, 
        f'edge_{sketch_view_index // SKETCH_PER_VIEW}_{sketch_view_index % SKETCH_PER_VIEW}.png')).convert('RGBA')
    
    img = self.to_tensor(img)
    pm = self.projection_list[sketch_view_index // SKETCH_PER_VIEW]
    projection_matrix = torch.from_numpy(np.expand_dims(pm, axis=0))
    return img, projection_matrix, sketch_view_index // SKETCH_PER_VIEW

  def __call__(self, uid):
    return self.random_load_image(uid)

def collate_func(batch):
  output = ocnn.dataset.CollateBatch(merge_points=False)(batch)

  if 'pos' in output:
    bi = [torch.ones(pos.size(0), 1) * i for i, pos in enumerate(output['pos'])]
    batch_idx = torch.cat(bi, dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)

  for key in ['grad', 'sdf', 'occu', 'weight', 'color', 'image', 'projection_matrix']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)

  return output


def get_shapenet_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_func
