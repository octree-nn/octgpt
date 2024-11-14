import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ocnn.octree import Octree, Points


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='downsample_points')
parser.add_argument('--input_folder', required=True, type=str)
parser.add_argument('--output_folder', required=True, type=str)
parser.add_argument('--filelist', required=True, type=str)
parser.add_argument('--octree_depth', type=int, default=8)
args = parser.parse_args()


def downsample_points():
  r''' Downsample the point clouds according to the octrees. '''

  with open(args.filelist, 'r') as fid:
    lines = fid.readlines()
  filenames = [line.split()[0] for line in lines]

  points_scale = 0.5
  for filename in tqdm(filenames, ncols=80):
    filename_in = os.path.join(args.input_folder, filename, 'pointcloud.npz')
    filename_out = os.path.join(args.output_folder, filename, 'pointcloud.npz')

    folder_out = os.path.dirname(filename_out)
    if not os.path.exists(folder_out):
      os.makedirs(folder_out)

    # get the input
    sample_in = np.load(filename_in)
    points = torch.from_numpy(sample_in['points']).float()
    normals = torch.from_numpy(sample_in['normals']).float()
    points /= points_scale    # scale to [-1.0, 1.0]

    # transform points to octree
    points_in = Points(points=points, normals=normals)
    points_in.clip(min=-1, max=1)
    octree_in = Octree(depth=args.octree_depth, full_depth=3)
    octree_in.build_octree(points_in)

    # transform octree to points
    points_out = octree_in.to_points()
    points_out.points *= points_scale  # scale to [-0.5, 0.5]
    np.savez(filename_out,
             points=points_out.points.numpy().astype(np.float16),
             normals=points_out.normals.numpy().astype(np.float16),)


def split_sdf():
  r''' Split the SDF files into multiple smaller files. '''

  with open(args.filelist, 'r') as fid:
    lines = fid.readlines()
  filenames = [line.split()[0] for line in lines]

  sdf_file_num = 8
  for filename in tqdm(filenames, ncols=80):
    filename_in = os.path.join(args.input_folder, filename, 'sdf.npz')
    filename_out = os.path.join(args.output_folder, filename, 'sdf')

    folder_out = os.path.dirname(filename_out)
    if not os.path.exists(folder_out):
      os.makedirs(folder_out)

    # get the input
    sample_in = np.load(filename_in)
    points = sample_in['points']
    grad = sample_in['grad']
    sdf = sample_in['sdf']

    # random permutation
    idx = np.random.permutation(points.shape[0])
    num = int(points.shape[0] / sdf_file_num + 0.5)
    for i in range(sdf_file_num):
      rng = idx[range(i * num, min((i + 1) * num, points.shape[0]))]
      np.savez(filename_out + f'_{i}.npz',
               points=points[rng], grad=grad[rng], sdf=sdf[rng])


if __name__ == '__main__':
  eval(args.run + '()')
