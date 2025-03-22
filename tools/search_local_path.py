import pandas as pd
import os
import glob
from tqdm import tqdm
from collections import defaultdict

def get_local_path_objaverse_github():
  load_path = f'data/Objaverse/ObjaverseXL_github'
  metadata_path = 'data/Objaverse/ObjaverseXL_github/metadata.csv'
  metadata = pd.read_csv(metadata_path)
  if "local_path" not in metadata.columns:
    metadata["local_path"] = None
  
  lost_data = []
  suffix_count = defaultdict(int)

  for i in tqdm(range(len(metadata))):
    filename = metadata.loc[i, "sha256"]
    file_identifier = metadata.loc[i, "file_identifier"]
    suffix = file_identifier.split(".")[-1].lower()
    suffix_count[suffix] += 1
    local_path_choice = [f"raw/{suffix}/{filename}.{suffix}"]
    if suffix == "fbx":
      local_path_choice = [f"raw/{suffix}_to_glb/{filename}.glb"]
    elif suffix == "gltf":
      local_path_choice.insert(0, f"raw/{suffix}_to_glb/{filename}.glb")
    local_path = None
    for path in local_path_choice:
      if os.path.exists(os.path.join(load_path, path)):
        local_path = path
        break
    if local_path is None:
      print(f"{filename} not found")
      lost_data.append(local_path_choice)
      continue
    metadata.loc[i, "local_path"] = local_path
  metadata.to_csv("data/Objaverse/filelist/ObjaverseXL_github.csv", index=False)

def get_objaverse_filelist():
  load_path = f'data/Objaverse/ObjaverseXL_sketchfab/repair'
  metadata_path = 'data/Objaverse/filelist/ObjaverseXL_sketchfab.csv'
  metadata = pd.read_csv(metadata_path)
  filelist = []
  for filename in metadata['sha256']:
    if os.path.exists(os.path.join(load_path, f"{filename}/sdf.npz")) and os.path.exists(os.path.join(load_path, f"{filename}/pointcloud.npz")):
      filelist.append(filename)

  with open("data/Objaverse/filelist/ObjaverseXL_sketchfab.txt", "w") as f:
    f.writelines([f"{filename}\n" for filename in filelist])

if __name__ == "__main__":
  # get_local_path_objaverse_github()
  get_objaverse_filelist()
