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
from models.mar import MAR, MAREncoderDecoder
from models.condition import ImageEncoder
from datasets import get_shapenet_dataset
from datasets.shapenet_utils import snc_synth_id_to_label_5, category_5_to_num
from tqdm import tqdm
import copy
import cv2


class OctarSolver(Solver):

  def __init__(self, FLAGS, is_master=True):
    super().__init__(FLAGS, is_master)
    self.depth = FLAGS.MODEL.depth
    self.depth_stop = FLAGS.MODEL.depth_stop
    self.full_depth = FLAGS.MODEL.full_depth
    self.condition_type = FLAGS.MODEL.GPT.condition_type

  def get_model(self, flags):
    if flags.model_name == "MAR":
      model = MAR(vqvae_config=flags.VQVAE, **flags.GPT)
    elif flags.model_name == "MAREncoderDecoder":
      model = MAREncoderDecoder(vqvae_config=flags.VQVAE, **flags.GPT)
    else:
      raise NotImplementedError("Model not implemented")

    vqvae = builder.build_vae_model(flags.VQVAE)
    model.cuda(device=self.device)
    vqvae.cuda(device=self.device)
    utils.set_requires_grad(model, True)
    utils.set_requires_grad(vqvae, False)

    if self.condition_type == "image":
      self.cond_enc = ImageEncoder(flags.GPT.condition_encoder)
      self.cond_enc.cuda(device=self.device)
      self.cond_enc.eval()
      utils.set_requires_grad(self.cond_enc, False)

    # load the pretrained vqvae
    checkpoint = torch.load(flags.vqvae_ckpt, weights_only=True, map_location="cuda")
    vqvae.load_state_dict(checkpoint)
    print("Load VQVAE from", flags.vqvae_ckpt)

    self.vqvae_module = vqvae
    self.model_module = model
    return model

  def get_dataset(self, flags):
    return get_shapenet_dataset(flags)

  def config_optimizer(self):
    super().config_optimizer()

  def batch_to_cuda(self, batch):
    keys = ['octree', 'octree_in', 'octree_gt', 'pos', 'sdf',
            'grad', 'weight', 'occu', 'color']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()

    if self.condition_type == "None":
      batch['condition'] = None
    elif self.condition_type == "category":
      label = [snc_synth_id_to_label_5[filename.split("/")[0]]
               for filename in batch['filename']]
      batch['condition'] = torch.tensor(label, device=self.device)
    elif self.condition_type == "image":
      images = batch['image']
      cond = self.cond_enc(images, device=self.device)
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
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    self.model.eval()
    category = self.FLAGS.DATA.test.filelist.split("_")[-1][:-4]
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
      # self.generate_vq_step(index)

  def export_results(self, octree_out, index, vq_code=None, image=None):
    # export the octree
    for d in range(self.full_depth + 1, self.depth_stop + 1):
      utils.export_octree(octree_out, d, os.path.join(
          self.logdir, f'results/octree_depth{d}'), index=index)

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
        os.path.join(self.logdir, f"results/{index}.obj"),
        size=self.FLAGS.SOLVER.resolution,
        bbmin=-self.FLAGS.SOLVER.sdf_scale,
        bbmax=self.FLAGS.SOLVER.sdf_scale,
        mesh_scale=self.FLAGS.DATA.test.points_scale,
        save_sdf=self.FLAGS.SOLVER.save_sdf)
    # Save the image
    if image is not None:
      os.makedirs(os.path.join(self.logdir, "results/images"), exist_ok=True)
      image[0].save(os.path.join(self.logdir, f"results/images/{index}.png"))

  @torch.no_grad()
  def generate_step(self, index):
    # forward the model
    batch = next(self.test_iter)
    self.batch_to_cuda(batch)
    octree_out = ocnn.octree.init_octree(
        self.depth, self.full_depth, self.FLAGS.DATA.test.batch_size, self.device)
    with torch.autocast("cuda", enabled=self.use_amp):
      octree_out, vq_code = self.model_module.generate(
          octree=octree_out,
          depth_low=self.full_depth, depth_high=self.depth_stop,
          vqvae=self.vqvae_module, condition=batch['condition'])

    self.export_results(octree_out, index, vq_code, batch['image'] if 'image' in batch else None)

  @torch.no_grad()
  def generate_vq_step(self, index):
    # forward the model
    batch = next(self.test_iter)
    self.batch_to_cuda(batch)
    octree_in = batch['octree_gt']
    split_seq = utils.octree2seq(octree_in, self.full_depth, self.depth_stop)
    with torch.autocast("cuda", enabled=self.use_amp):
      octree_out, vq_code = self.model_module.generate(
          octree=octree_in,
          depth_low=self.full_depth, depth_high=self.depth_stop,
          token_embeddings=self.model_module.split_emb(split_seq),
          vqvae=self.vqvae_module, condition=batch['condition'])

    vq_indices = self.vqvae_module.quantizer(vq_code)[1]
    gt_vq_code = self.vqvae_module.extract_code(octree_in)
    gt_indices = self.vqvae_module.quantizer(gt_vq_code)[1]

    print(
        f"{torch.where(vq_indices != gt_indices)[0].shape}/{vq_indices.numel()} indices are different")
    self.export_results(octree_in, index + 1, gt_vq_code)
    self.export_results(octree_out, index, vq_code)
  
  def load_checkpoint(self):
    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      # If ckpt is empty, then get the latest checkpoint from ckpt_dir
      if not os.path.exists(self.ckpt_dir):
        return
      ckpts = sorted(os.listdir(self.ckpt_dir))
      ckpts = [ck for ck in ckpts if ck.endswith('solver.tar')]
      if len(ckpts) > 0:
        ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
    if not ckpt:
      return  # return if ckpt is still empty

    # load trained model
    # check: map_location = {'cuda:0' : 'cuda:%d' % self.rank}
    trained_dict = torch.load(ckpt, map_location='cuda')
    if ckpt.endswith('.solver.tar'):
      model_dict = trained_dict['model_dict']
      self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
      if self.optimizer:
        self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
      if self.scaler and 'scaler_dict' in trained_dict:
        self.scaler.load_state_dict(trained_dict['scaler_dict'])
    else:
      model_dict = trained_dict
    model = self.model.module if self.world_size > 1 else self.model
    if self.FLAGS.SOLVER.expand_ckpt:
      model_dict = expand_checkpoint(model, model_dict)
    model.load_state_dict(model_dict)

    # print messages
    if self.is_master:
      tqdm.write('Load the checkpoint: %s' % ckpt)
      tqdm.write('The start_epoch is %d' % self.start_epoch)


if __name__ == '__main__':
  OctarSolver.main()
