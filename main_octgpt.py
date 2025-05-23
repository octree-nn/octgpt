import os
import torch
import ocnn
import time
from thsolver import Solver
from thsolver.tracker import AverageTracker
from ognn.octreed import OctreeD
from utils import utils, builder
from utils.expand_ckpt import expand_checkpoint
from utils.distributed import get_rank
from models.octgpt import OctGPT
from models.condition import ImageEncoder, TextEncoder
from datasets.shapenet_utils import snc_synth_id_to_label_5, category_5_to_num, snc_synth_id_to_label_13, snc_category_to_synth_id_all
from tqdm import tqdm
import copy
import cv2


class OctGPTSolver(Solver):

  def __init__(self, FLAGS, is_master=True):
    super().__init__(FLAGS, is_master)
    self.depth = FLAGS.MODEL.depth
    self.depth_stop = FLAGS.MODEL.depth_stop
    self.full_depth = FLAGS.MODEL.full_depth
    self.condition_type = FLAGS.MODEL.OctGPT.condition_type

  def get_model(self, flags):
    if flags.model_name == "OctGPT":
      model = OctGPT(vqvae_config=flags.VQVAE, **flags.OctGPT)
    else:
      raise NotImplementedError("Model not implemented")

    vqvae = builder.build_vae_model(flags.VQVAE)
    model.cuda(device=self.device)
    vqvae.cuda(device=self.device)
    utils.set_requires_grad(model, True)
    utils.set_requires_grad(vqvae, False)

    if self.condition_type == "image":
      self.cond_enc = ImageEncoder(flags.OctGPT.condition_encoder)
    elif self.condition_type == "text":
      self.cond_enc = TextEncoder(flags.OctGPT.condition_encoder)
    else:
      self.cond_enc = None

    if self.cond_enc:
      self.cond_enc.cuda(device=self.device)
      self.cond_enc.eval()
      utils.set_requires_grad(self.cond_enc, False)

    # load the pretrained vqvae
    checkpoint = torch.load(
        flags.vqvae_ckpt, weights_only=True, map_location="cuda")
    vqvae.load_state_dict(checkpoint)
    print("Load VQVAE from", flags.vqvae_ckpt)

    self.vqvae_module = vqvae
    self.model_module = model
    return model

  def get_dataset(self, flags):
    return builder.build_dataset(flags)

  def config_optimizer(self):
    super().config_optimizer()

  def batch_to_cuda(self, batch):
    keys = ['octree', 'octree_in', 'octree_gt', 'pos', 'sdf',
            'grad', 'weight', 'occu', 'color']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()

    if self.condition_type == "none":
      batch['condition'] = None
    elif self.condition_type == "category":
      if self.FLAGS.MODEL.OctGPT.num_classes == 5:
        id_to_label = snc_synth_id_to_label_5
      elif self.FLAGS.MODEL.OctGPT.num_classes == 13:
        id_to_label = snc_synth_id_to_label_13

      if "category" in batch:
        label = [id_to_label[snc_category_to_synth_id_all[cat]]
                 for cat in batch['category']]
      else:
        label = [id_to_label[filename.split("/")[0]]
                 for filename in batch['filename']]
      batch['condition'] = torch.tensor(label, device=self.device)

    elif self.condition_type == "image":
      images = batch['image']
      cond = self.cond_enc(images, device=self.device)
      batch['condition'] = cond
    elif self.condition_type == "text":
      text = batch['text']
      cond = self.cond_enc(text, device=self.device)
      batch['condition'] = cond
    else:
      raise NotImplementedError("Condition type not implemented")

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    octree_in = batch['octree_gt']

    split_seq = utils.octree2seq(octree_in, self.full_depth, self.depth_stop)
    output = self.model(
        octree_in=octree_in, depth_low=self.full_depth, split=split_seq,
        depth_high=self.depth_stop, vqvae=self.vqvae_module,
        condition=batch['condition'])
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    with torch.no_grad():
      output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def test_epoch(self, epoch):
    if epoch % 5 != 0:
      return
    super().test_epoch(epoch)
    # generate the mesh
    self.generate_step(epoch + get_rank())

  def generate(self):
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    # self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    self.model.eval()
    category = self.FLAGS.DATA.test.get("category")

    num_meshes = category_5_to_num[category] if category in category_5_to_num else 10000
    mesh_indices = []
    for i in range(num_meshes):
      mesh_path = os.path.join(self.logdir, f"results/{i}.obj")
      if not os.path.exists(mesh_path):
        mesh_indices.append(i)

    for iter in tqdm(range(0, 10000), ncols=80):
      if self.world_size * iter + get_rank() >= len(mesh_indices):
        break
      index = mesh_indices[self.world_size * iter + get_rank()]
      self.generate_step(index)

  def export_results(self, octree_out, index, vq_code=None, image=None, text=None, output_dir="results"):
    # export the octree
    for d in range(self.full_depth + 1, self.depth_stop + 1):
      utils.export_octree(octree_out, d, os.path.join(
          self.logdir, f'{output_dir}/octree_depth{d}'), index=index)

    # decode the octree
    for d in range(self.depth_stop, self.depth):
      split_zero_d = torch.zeros(
          octree_out.nnum[d], device=octree_out.device).long()
      octree_out.octree_split(split_zero_d, d)
      octree_out.octree_grow(d + 1)
    doctree_out = OctreeD(octree_out)
    with torch.no_grad():
      output = self.vqvae_module.decode_code(
          vq_code, self.depth_stop, doctree_out,
          copy.deepcopy(doctree_out), update_octree=True)

    # extract the mesh
    utils.create_mesh(
        output['neural_mpu'],
        os.path.join(self.logdir, f"{output_dir}/{index}.obj"),
        size=self.FLAGS.SOLVER.resolution,
        level=0.002, clean=True,
        bbmin=-self.FLAGS.SOLVER.sdf_scale,
        bbmax=self.FLAGS.SOLVER.sdf_scale,
        mesh_scale=self.FLAGS.DATA.test.points_scale,
        save_sdf=self.FLAGS.SOLVER.save_sdf)
    # Save the image
    if image is not None:
      os.makedirs(os.path.join(
          self.logdir, f"{output_dir}/images"), exist_ok=True)
      image[0].save(os.path.join(
          self.logdir, f"{output_dir}/images/{index}.png"))
    # Save the text:
    if text is not None:
      os.makedirs(os.path.join(
          self.logdir, f"{output_dir}/text"), exist_ok=True)
      with open(os.path.join(self.logdir, f"{output_dir}/text/{index}.txt"), "w") as f:
        f.write(text[0] + '\n')

  @torch.no_grad()
  def generate_step(self, index):
    # forward the model
    batch = {}
    if self.condition_type == "category" and self.FLAGS.DATA.test.get("category"):
      batch['category'] = [self.FLAGS.DATA.test.category] * \
          self.FLAGS.DATA.test.batch_size
      self.batch_to_cuda(batch)
      condition = batch['condition']
    elif self.condition_type == "text" and self.FLAGS.DATA.test.get("text_prompt"):
      batch['text'] = [self.FLAGS.DATA.test.text_prompt] * \
          self.FLAGS.DATA.test.batch_size
      self.batch_to_cuda(batch)
      condition = batch['condition']
    else:
      condition = None

    octree_out = ocnn.octree.init_octree(
        self.depth, self.full_depth, self.FLAGS.DATA.test.batch_size, self.device)
    with torch.autocast("cuda", enabled=self.use_amp):
      octree_out, vq_code = self.model_module.generate(
          octree=octree_out, depth_low=self.full_depth, depth_high=self.depth_stop,
          vqvae=self.vqvae_module, condition=condition)

    self.export_results(
        octree_out, index, vq_code, output_dir=f"results",
        # image=batch['image'] if self.condition_type == 'image' else None,
        text=batch['text'] if self.condition_type == 'text' else None)


if __name__ == '__main__':
  OctGPTSolver.main()
