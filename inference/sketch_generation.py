import torch
import yaml
from utils.builder import build_vae_model
from models.condition import ImageEncoder
from models.octgpt import OctGPT
from thsolver.config import parse_args
import sys 
import matplotlib.pyplot as plt
import ocnn
from PIL import Image
import os
import copy
from ognn.octreed import OctreeD
from utils import utils
torch.cuda.set_device(0)

# log_path = '/mnt/sdc/wangrh/workspace/OctAR-solver/logs/sketch/airplane_p1024_d8'
log_path = '/mnt/sdc/wangrh/workspace/OctAR-solver/logs/image/car_p1024_d8'

sys.argv = ['']  # Reset sys.argv
sys.argv.extend(['--config', log_path + '/all_configs.yaml'])
flags = parse_args(backup=False)

device = 'cuda'

model = OctGPT(vqvae_config=flags.MODEL.VQVAE, **flags.MODEL.OctGPT)
vqvae = build_vae_model(flags.MODEL.VQVAE)
sketch_encoder = ImageEncoder(flags.MODEL.OctGPT.condition_encoder)

vqvae_checkpoint = torch.load(flags.MODEL.vqvae_ckpt, weights_only=True, map_location="cpu")
vqvae.load_state_dict(vqvae_checkpoint)
print("Load VQVAE from", flags.MODEL.vqvae_ckpt)

ar_checkpoint = log_path + '/checkpoints/00118.model.pth'
model_checkpoint = torch.load(ar_checkpoint, map_location="cpu")
model.load_state_dict(model_checkpoint)
print("Load MAR from", ar_checkpoint)

model = model.to(device)
vqvae = vqvae.to(device)
sketch_encoder = sketch_encoder.to(device)

depth = flags.DATA.test.depth
full_depth = flags.DATA.test.full_depth
depth_stop = flags.MODEL.depth_stop

def generate_by_sketch(sketch_path, sketchname='default'):
    sketch = Image.open(sketch_path)

    with torch.no_grad():
        condition = sketch_encoder(sketch, device=device)
        octree_out = ocnn.octree.init_octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=1,
            device=device,
        )
        with torch.autocast('cuda', enabled=flags.SOLVER.use_amp):
            octree_out, vq_code = model.generate(
                octree=octree_out,
                depth_low=full_depth,
                depth_high=depth_stop,
                vqvae=vqvae,
                condition=condition,
                cfg_scale=None
            )

    export_path = f'results-inference/{sketchname}'

    index = 'output'

    # Export octrees
    for d in range(full_depth+1, depth_stop+1):
        utils.export_octree(octree_out, d, os.path.join(
            log_path ,export_path), index=f'octree_{d}')

    # Decode the mesh
    for d in range(depth_stop, depth):
        split_zero_d = torch.zeros(
            octree_out.nnum[d], device=octree_out.device).long()
        octree_out.octree_split(split_zero_d, d)
        octree_out.octree_grow(d + 1)
    doctree_out = OctreeD(octree_out)
    with torch.no_grad():
        output = vqvae.decode_code(
            vq_code, depth_stop, doctree_out,
            copy.deepcopy(doctree_out), update_octree=True)

    # extract the mesh
    utils.create_mesh(
        output['neural_mpu'],
        os.path.join(log_path, export_path, f"output.obj"),
        size=flags.SOLVER.resolution,
        level=0.002, clean=True,
        bbmin=-flags.SOLVER.sdf_scale,
        bbmax=flags.SOLVER.sdf_scale,
        mesh_scale=flags.DATA.test.points_scale,
        save_sdf=flags.SOLVER.save_sdf)

    # Save the sketch image
    sketch.save(os.path.join(log_path, export_path, f"input.png"))

# sketch_paths = [f'/mnt/sdc/wangrh/workspace/OctAR-solver/logs/sketch/airplane_p1024_d8/results/images/{i}.png' for i in range(1, 50)]
sketch_path = '/mnt/sdc/weist/data/ShapeNet/fid_images/car/view_0'
sketch_paths = os.listdir(sketch_path)
sketch_paths = [os.path.join(sketch_path, i) for i in sketch_paths]
import random
random.shuffle(sketch_paths)
sketch_paths[:5]

for i, sketch_path in enumerate(sketch_paths[:400]):
    print(f"Generating {i+1}/{len(sketch_paths)}")
    try:
        generate_by_sketch(sketch_path, f"sketch_{i+1}")
    except:
        pass
    print("Done")