import os
import torch
import ocnn
import ognn
import torch.nn.functional as F
from thsolver import Solver
from ognn.octreed import OctreeD

from utils import utils, builder


class VAESolver(Solver):

  def get_model(self, flags):
    return builder.build_vae_model(flags)

  def get_dataset(self, flags):
    return builder.build_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = ['octree', 'octree_in', 'octree_gt', 'pos', 'sdf',
            'grad', 'weight', 'occu', 'color']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()
    batch['pos'].requires_grad_()

  def compute_scene_loss(self, batch, model_out):
    wo = [1.0] * 8 + [0.1] * 3  # lower weights for deep layers
    output = ognn.loss.synthetic_room_loss(batch, model_out, wo=wo)
    output['vae_loss'] = self.FLAGS.LOSS.vae_weight * model_out['vae_loss']
    return output

  def compute_shape_loss(self, batch, model_out, reg_loss_type):
    wo = [1.0] * 8 + [0.1] * 3  # lower weights for deep layers
    output = ognn.loss.shapenet_loss(
        batch, model_out, reg_loss_type=reg_loss_type, wo=wo)
    output['vae_loss'] = self.FLAGS.LOSS.vae_weight * model_out['vae_loss']
    return output

  def compute_loss(self, batch, model_out, reg_loss_type):
    if self.FLAGS.LOSS.name == 'shape':
      return self.compute_shape_loss(batch, model_out, reg_loss_type)
    elif self.FLAGS.LOSS.name == 'room':
      return self.compute_scene_loss(batch, model_out)
    else:
      raise ValueError('Unsupported loss type')

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    octree_in = batch['octree_in']
    octree_gt = OctreeD(batch['octree_gt'])
    model_out = self.model(octree_in, octree_gt, batch['pos'])

    output = self.compute_loss(batch, model_out, self.FLAGS.LOSS.reg_loss_type)
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def test_epoch(self, epoch):
    torch.cuda.empty_cache()
    # set test_every_epoch to 1, so that we can save checkpoints every epoch
    # Test the model every 5 epochs
    if epoch % 5 == 0:
      super().test_epoch(epoch)

    # Generate one mesh for eval
    if self.is_master:
      batch = next(self.test_iter)
      filename = batch['filename'][0]
      batch['filename'][0] = os.path.join("results", f"{epoch}/{filename}")
      self.eval_step(batch)

  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree_in'].cuda()
    octree_out = OctreeD(octree_in)  # initialize
    # octree_out = self._init_octree_out(octree_in)
    output = self.model(octree_in, octree_out, update_octree=True)

    # extract the mesh
    flags = self.FLAGS.DATA.test
    filename = self._extract_filename(batch)
    bbmin, bbmax = self._get_bbox(batch)
    utils.create_mesh(
        output['neural_mpu'], filename, size=flags.resolution,
        bbmin=bbmin, bbmax=bbmax, mesh_scale=flags.points_scale,
        save_sdf=flags.save_sdf, clean=True, level=0.002)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    points = batch['points_in'][0]
    points.points *= flags.points_scale
    utils.points2ply(filename, points)

  def _extract_filename(self, batch):
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1: filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, "results", filename + '.obj')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder): os.makedirs(folder)
    return filename

  def _init_octree_out(self, octree_in):
    full_depth = octree_in.full_depth  # grow octree to full_depth
    octree_out = ocnn.octree.init_octree(
        full_depth, full_depth, octree_in.batch_size, octree_in.device)
    return OctreeD(octree_out, full_depth)

  def _get_bbox(self, batch):
    if 'bbox' in batch:
      bbox = batch['bbox'][0].numpy()
      bbmin, bbmax = bbox[:3], bbox[3:]
    else:
      sdf_scale = self.FLAGS.DATA.test.sdf_scale
      bbmin, bbmax = -sdf_scale, sdf_scale
    return bbmin, bbmax


if __name__ == '__main__':
  VAESolver.main()
