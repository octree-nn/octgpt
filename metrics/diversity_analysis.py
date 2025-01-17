import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.evaluation_metrics import distChamfer
from utils.utils import get_filenames
import trimesh
import numpy as np
import torch
import pickle
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"


def scale_to_unit_sphere_pc(points):
  centroid = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
  points -= centroid
  distances = np.linalg.norm(points, axis=1)
  points /= np.max(distances)
  return points


def compute_metrics(sample_pcs, ref_pcs, batch_size):

  N_ref = ref_pcs.shape[0]
  cd_lst = []
  for ref_b_start in range(0, N_ref, batch_size):
    ref_b_end = min(N_ref, ref_b_start + batch_size)
    ref_batch = ref_pcs[ref_b_start:ref_b_end]

    batch_size_ref = ref_batch.size(0)
    sample_batch_exp = sample_pcs.view(1, -1, 3).expand(batch_size_ref, -1, -1)
    sample_batch_exp = sample_batch_exp.contiguous()
    dl, dr = distChamfer(sample_batch_exp, ref_batch)
    cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
    cd_lst.append(cd)

  cd_lst = torch.cat(cd_lst, dim=1)

  return cd_lst


num_samples = 2048
topk = 5
category = "car"
mesh_dir = '/home/zhoucz/rendering/ar/uncond/car'
filelist_dir = "data/ShapeNet/filelist"
pointcloud_dir = "data/ShapeNet/dataset_new"
collect_dir = "data/ShapeNet/pointcloud_2048"
similar_dir = "mytools/similar"

def collect_pointclouds():
  pointcloud_dict = {}
  filenames = get_filenames(os.path.join(filelist_dir, f"train_{category}.txt"))
  filename_collect = os.path.join(collect_dir, f"{category}.pkl")
  if os.path.exists(filename_collect):
    print(f"File {filename_collect} already exists.")
    return
  for filename in tqdm(filenames):
    filename_pointcloud = os.path.join(
        pointcloud_dir, filename, "pointcloud.npz")
    raw = np.load(filename_pointcloud)
    points = raw['points']
    points = scale_to_unit_sphere_pc(points)
    points = np.random.permutation(points)[:num_samples]
    pointcloud_dict[filename] = points

  os.makedirs(collect_dir, exist_ok=True)
  with open(filename_collect, 'wb') as file:
    pickle.dump(pointcloud_dict, file)


def calc_diversity(find_topk=False):
  min_cd_list = []
  with open(os.path.join(collect_dir, f"{category}.pkl"), 'rb') as file:
    raw = pickle.load(file)
  ref_key, ref_pcs = list(raw.keys()), list(raw.values())
  ref_pcs = np.stack(ref_pcs)
  ref_pcs = torch.from_numpy(ref_pcs).cuda().to(torch.float32)

  for filename in tqdm(os.listdir(mesh_dir)):
    if not filename.endswith('.obj'):
      continue
    mesh_path = os.path.join(mesh_dir, filename)

    mesh = trimesh.load(mesh_path, force='mesh')
    sample_pcs, idx = trimesh.sample.sample_surface(mesh, num_samples)
    sample_pcs = scale_to_unit_sphere_pc(sample_pcs)

    sample_pcs = torch.from_numpy(sample_pcs).cuda().to(torch.float32)
    cd_list = compute_metrics(sample_pcs, ref_pcs, batch_size=64).squeeze(0)
    min_cd = cd_list.min().item()
    min_cd_list.append(min_cd)
    if find_topk:
      print(filename)
      topk_values, topk_indices = torch.topk(cd_list, topk, largest=False)
      for i in range(topk):
        similar_key = ref_key[topk_indices[i].long().item()]
        print(f"Top {i+1} CD: {topk_values[i]} filename: {similar_key}")
        filename_similar = os.path.join(similar_dir, f"{category}/{filename[:-4]}/", f"top{i}.obj")
        os.makedirs(os.path.dirname(filename_similar), exist_ok=True)
        shutil.copy2(f"data/ShapeNet/mesh_256/{similar_key}.obj", filename_similar)

  min_cd_list = np.array(min_cd_list)
  np.save(os.path.join(mesh_dir, f"min_cd.npy"), min_cd_list)

def plot_hist():
  min_cd_list = np.load(os.path.join(mesh_dir, f"min_cd.npy"))
  min_cd_list *= 1000
  
  bins = [0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40]

  plt.hist(min_cd_list, bins=bins)
  plt.xlabel('Chamfer Distance')
  plt.ylabel('Count')
  plt.title('Histogram')
  plt.savefig(os.path.join(mesh_dir, f"min_cd_hist.png"))
  plt.close()

if __name__ == "__main__":
  # collect_pointclouds()
  calc_diversity()
  # plot_hist()