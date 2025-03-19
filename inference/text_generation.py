import torch
import yaml
import sys 
import ocnn
import os
import copy
from ognn.octreed import OctreeD
from thsolver.config import parse_args
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils.builder import build_vae_model
from models.condition import TextEncoder 
from models.octgpt import OctGPT

torch.cuda.set_device(3)
log_path = 'logs/objaverse/mar_text_bvflip5_expand1152/'

sys.argv = ['']  # Reset sys.argv
sys.argv.extend(['--config', os.path.join(log_path, 'all_configs.yaml')])
flags = parse_args(backup=False)

device = 'cuda'

model = OctGPT(vqvae_config=flags.MODEL.VQVAE, **flags.MODEL.OctGPT)
vqvae = build_vae_model(flags.MODEL.VQVAE)
text_encoder = TextEncoder(flags.MODEL.OctGPT.condition_encoder)

vqvae_checkpoint = torch.load(flags.MODEL.vqvae_ckpt, weights_only=True, map_location="cpu")
vqvae.load_state_dict(vqvae_checkpoint)
print("Load VQVAE from", flags.MODEL.vqvae_ckpt)

ar_checkpoint = 'logs/objaverse/mar_text_bvflip5_expand1152/checkpoints/00027.model.pth'
model_checkpoint = torch.load(ar_checkpoint, map_location="cpu")
model.load_state_dict(model_checkpoint)
print("Load MAR from", ar_checkpoint)

model = model.to(device)
vqvae = vqvae.to(device)
text_encoder = text_encoder.to(device)

text = '3D palm tree model.'
export_path = f'results-inference/{text}'
os.makedirs(os.path.join(log_path, export_path), exist_ok=True)
# Save the text:
with open(os.path.join(log_path, export_path, f"input.txt"), "w") as f:
    f.write(text + '\n')

depth = flags.DATA.test.depth
full_depth = flags.DATA.test.full_depth
depth_stop = flags.MODEL.depth_stop
model.num_iters = [64, 128, 128, 256]
num_gen = 4

for i in range(num_gen):
    with torch.no_grad():
        condition = text_encoder(text, device=device)
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
            )
    # Export octrees
    for d in range(full_depth+1, depth_stop+1):
        utils.export_octree(octree_out, d, os.path.join(
            log_path, export_path), index=f'octree_{d}')

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
    
    