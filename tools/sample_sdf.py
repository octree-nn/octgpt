import trimesh
import mesh2sdf
import os
import sys
import ocnn
import numpy as np
import torch
import diso
import mcubes
import traceback
from functools import partial
import torchcumesh2sdf
import multiprocessing as mp
import objaverse.xl as oxl
from tqdm import tqdm
import pandas as pd
import glob
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import scale_to_unit_cube

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="ShapeNet", type=str)
parser.add_argument('--mode', default="cpu", type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--mesh_scale', type=float, default=0.8)
# parser.add_argument('--device_list', type=str, default="0,1,2,3")
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--level', type=float, default=1/256)
parser.add_argument('--band', type=float, default=0.05)
parser.add_argument('--num_samples', type=int, default=200000)
parser.add_argument('--num_processes', type=int, default=32)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--full_depth', type=int, default=4)
args = parser.parse_args()

args.size = 2 ** args.depth
args.level = 1 / args.size
if args.debug:
  args.num_processes = 1
else:
  args.num_processes = 32


def check_folder(filenames: list):
  for filename in filenames:
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
      os.makedirs(folder)

def sample_pts(mesh, filename_pts):
  points, idx = trimesh.sample.sample_surface(mesh, args.num_samples)
  normals = mesh.face_normals[idx]
  np.savez(filename_pts, points=points.astype(np.float16),
          normals=normals.astype(np.float16))
  return {'points': points, 'normals': normals}


def sample_sdf(sdf, filename_out, pts=None):
  # constants
  depth, full_depth = args.depth, args.full_depth
  sample_num = 4  # number of samples in each octree node 也就是文中说的在每个八叉树的节点，采4个点并计算对应的sdf值。
  grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
  grid = torch.tensor(grid, device=sdf.device)

  points = pts['points'].astype(np.float32)
  normals = pts['normals'].astype(np.float32)

  # build octree 这里构建八叉树时输入的点云是[-1,1]内连续的，也就是说不受分辨率的影响
  points = ocnn.octree.Points(torch.from_numpy(
      points), torch.from_numpy(normals)).to(sdf.device)
  octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
  octree.build_octree(points)

  # sample points and grads according to the xyz
  xyzs, grads, sdfs, valids = [], [], [], []
  for d in range(full_depth, depth + 1):
    xyzb = octree.xyzb(d)
    x, y, z, b = xyzb
    xyz = torch.stack((x, y, z), dim=1).float()

    # sample k points in each octree node
    xyz = xyz.unsqueeze(
        1) + torch.rand(xyz.shape[0], sample_num, 3, device=sdf.device)
    xyz = xyz.view(-1, 3)                  # (N, 3)
    # normalize to [0, 2^sdf_depth] 相当于将坐标放大到[0,128]，128是sdf采样的分辨率
    xyz = xyz * (args.size / (2 ** d))
    # remove out-of-bound points
    xyz = xyz[(xyz < (args.size - 1)).all(dim=1)]
    xyzs.append(xyz)

    # interpolate the sdf values
    xyzi = torch.floor(xyz)                # the integer part (N, 3)
    corners = xyzi.unsqueeze(1) + grid     # (N, 8, 3)
    coordsf = xyz.unsqueeze(1) - corners   # (N, 8, 3), in [-1.0, 1.0]
    weights = (1 - coordsf.abs()).prod(dim=-1)  # (N, 8, 1)
    corners = corners.long().view(-1, 3)
    x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
    s = sdf[x, y, z].view(-1, 8)
    sw = torch.sum(s * weights, dim=1)
    sdfs.append(sw)

    # test if sdf is in range
    valid = (s.abs() <= args.band).all(dim=1)
    valids.append(valid)

    # calc the gradient
    gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + \
        s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]  # noqa
    gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + \
        s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]  # noqa
    gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + \
        s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]  # noqa
    grad = torch.stack([gx, gy, gz], dim=-1)
    norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdims=True))
    grad = grad / (norm + 1.0e-8)
    grads.append(grad)

  # concat the results
  xyzs = torch.cat(xyzs, dim=0)
  points = (xyzs / (args.size/2) - 1)
  grads = torch.cat(grads, dim=0)
  sdfs = torch.cat(sdfs, dim=0)
  valids = torch.cat(valids, dim=0)

  # remove invalid points
  if args.mode == "cuda":
    # points = points[valids]
    # grads = grads[valids]
    # sdfs = sdfs[valids]
    sdfs[~valids] = (sdfs[~valids] > 0).float() * args.band
    grads[~valids] = 0.0

  # save results
  random_idx = torch.randperm(points.shape[0])[:min(400000, points.shape[0])]
  points = points[random_idx].cpu().numpy().astype(np.float16)
  grads = grads[random_idx].cpu().numpy().astype(np.float16)
  sdfs = sdfs[random_idx].cpu().numpy().astype(np.float16)
  np.savez(filename_out, points=points, grad=grads, sdf=sdfs)

  # visualize
  if args.debug:
    valids = valids[random_idx].cpu().numpy()
    surf_points = points[valids] - sdfs[valids].reshape(-1, 1) * grads[valids]
    pointcloud = trimesh.PointCloud(surf_points)
    pointcloud.export("mytools/sdf_surf.ply")
    pointcloud = trimesh.PointCloud(points)
    pointcloud.export("mytools/sdf.ply")


def get_sdf(mesh, filename_obj):
  vertices = mesh.vertices
  # run mesh2sdf
  voxel_sdf, mesh_new = mesh2sdf.compute(
      vertices, mesh.faces, args.size, fix=True, level=args.level, return_mesh=True)
  if args.debug:
    mesh_new.export(filename_obj)
  return mesh_new, voxel_sdf


def get_sdf_cu(mesh, device, filename_obj):
  tris = np.array(mesh.triangles, dtype=np.float32, subok=False)
  tris = torch.tensor(tris, dtype=torch.float32).to(device)
  tris = (tris + 1.0) / 2.0
  voxel_sdf = torchcumesh2sdf.get_sdf(tris, args.size, args.band, B=args.batch_size)
  if args.debug:
    vertices, faces = diso.DiffMC().to(device).forward(voxel_sdf, isovalue=args.level)
    mcubes.export_obj(vertices.cpu().numpy(),
                      faces.cpu().numpy(), filename_obj)
  torchcumesh2sdf.free_cached_memory()
  torch.cuda.empty_cache()
  return mesh, voxel_sdf

def process(index, filenames, load_paths, save_paths):
  filename = filenames[index]
  load_path = load_paths[index]
  save_path = save_paths[index]
  if not os.path.exists(load_path):
    print(f"{filename} not exists")
    return

  filename_input = load_path
  filename_obj = os.path.join(save_path, "mesh.obj")
  filename_pointcloud = os.path.join(save_path, 'pointcloud.npz')
  filename_sdf = os.path.join(save_path, 'sdf.npz')
  if os.path.exists(filename_sdf) and os.path.exists(filename_pointcloud):
    print(f"{filename} already exists")
    return

  try:
    mesh = trimesh.load(filename_input, force='mesh')
    mesh = scale_to_unit_cube(mesh)
    mesh.vertices *= args.mesh_scale
  except:
    # os.remove(filename_input)
    print(f"Trimesh load mesh {filename} error")
    return

  check_folder([filename_obj, filename_pointcloud, filename_sdf])
  # device = torch.device(f'cuda:{device_list[index % len(device_list)]}')
  device = torch.device('cuda:0')
  if args.mode == "cuda":
    mesh, voxel_sdf = get_sdf_cu(mesh, device, filename_obj)
  elif args.mode == "cpu":
    mesh, voxel_sdf = get_sdf(mesh, filename_obj)
    voxel_sdf = torch.tensor(voxel_sdf)
  pointcloud = sample_pts(mesh, filename_pointcloud)
  sample_sdf(voxel_sdf, filename_sdf, pointcloud)
  print(f"Mesh {index}/{len(filenames)} {filename} done")

def get_shapenet_path():
  load_path = f'data/ShapeNet/ShapeNetCore.v1'
  save_path = f'data/ShapeNet/datasets_256'
  existing_files = set(glob.glob(f"{save_path}/*/*.npz"))
  filelist_path = f'data/ShapeNet/filelist/im5.txt'
  with open(filelist_path, 'r') as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  load_paths, save_paths, filenames = [], [], []
  for line in lines:
    filename = line.split('/')[-1]
    if not isinstance(filename, str):
      continue
    if os.path.join(save_path, f"{filename}", "sdf.npz") in existing_files and \
       os.path.join(save_path, f"{filename}", "pointcloud.npz") in existing_files:
      continue
    filenames.append(filename)
    load_paths.append(os.path.join(load_path, line))
    save_paths.append(os.path.join(save_path, f"{filename}"))
  return filenames, load_paths, save_paths

def get_objaverse_path():
  load_path = f'data/Objaverse/ObjaverseXL_sketchfab'
  save_path = f'data/Objaverse/ObjaverseXL_sketchfab/datasets_512'
  existing_files = set(glob.glob(f"{save_path}/*/*.npz"))
  metadata_path = 'data/Objaverse/filelist/ObjaverseXL_sketchfab.csv'
  metadata = pd.read_csv(metadata_path)
  load_paths, save_paths, filenames = [], [], []
  for local_path, filename in zip(metadata['local_path'], metadata['sha256']):
    if not isinstance(local_path, str):
      continue
    if os.path.join(save_path, f"{filename}", "sdf.npz") in existing_files and \
       os.path.join(save_path, f"{filename}", "pointcloud.npz") in existing_files:
      continue
    filenames.append(filename)
    load_paths.append(os.path.join(load_path, local_path))
    save_paths.append(os.path.join(save_path, f"{filename}"))
  return filenames, load_paths, save_paths


if __name__ == "__main__":
  if args.dataset == "ShapeNet":
    filenames, load_paths, save_paths = get_shapenet_path()
  elif args.dataset == "Objaverse":
    filenames, load_paths, save_paths = get_objaverse_path()
  indices = list(range(len(filenames)))
  if args.num_processes > 1:
    func = partial(process, filenames=filenames, load_paths=load_paths, save_paths=save_paths)
    with mp.Pool(processes=args.num_processes) as pool:
      list(tqdm(pool.imap_unordered(func, indices), total=len(filenames)))
  else:
    for i in range(len(filenames)):
      process(i, filenames, load_paths, save_paths)
