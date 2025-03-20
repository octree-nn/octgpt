import pandas as pd
import os
import glob
from tqdm import tqdm
from collections import defaultdict

def get_local_path_Objaverse_github():
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


if __name__ == "__main__":
  get_local_path_Objaverse_github()
