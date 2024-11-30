import os
import csv
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', type=str, default='')
parser.add_argument('--filelist', type=str, default='')
parser.add_argument('--output_file', type=str, default='')
args = parser.parse_args()

root_folder = args.root_folder
filelist = args.filelist
output_file = args.output_file

# Read filelist
with open(filelist, 'r') as f:
  lines = f.readlines()
filenames = [line.strip() for line in lines]

# Get file sizes
file_info = []
for filename in tqdm(filenames):
  filename = os.path.join(root_folder, filename, 'pointcloud.npz')
  size = os.path.getsize(filename)
  pointcloud = np.load(filename)
  points = pointcloud['points']

  # print(f"Size of {filename}: {size} bytes")
  file_info.append({
      'path': filename,
      'size': round(size / (1024 * 1024), 2),  # Convert to MB
      'points': points.shape[0],
  })


# Write to CSV file
headers = ['path', 'size', 'points']
with open(output_file, 'w', newline='', encoding='utf-8') as f:
  writer = csv.DictWriter(f, fieldnames=headers)
  writer.writeheader()
  writer.writerows(file_info)
