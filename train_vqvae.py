import os
import torch
import ocnn
import ognn
from thsolver import Solver
from ognn.octreed import OctreeD

from utils import utils
# from models.vqvae import VQVAE
from models.vqvaev2 import VQVAE
from datasets import get_shapenet_vae_dataset


class VAESolver(Solver):

  def get_model(self, flags):
    return VQVAE(**flags)

  def get_dataset(self, flags):
    return get_shapenet_vae_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = ['octree', 'octree_in', 'octree_gt', 'pos', 'sdf',
            'grad', 'weight', 'occu', 'color']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()
    batch['pos'].requires_grad_()

  def compute_loss(self, batch, model_out):
    # octree loss
    output = ognn.loss.compute_octree_loss(
        model_out['logits'], model_out['octree_out'])

    # regression loss
    mpus = model_out['mpus']
    grads = ognn.loss.compute_mpu_gradients(mpus, batch['pos'])
    for d in mpus.keys():
      sdf = mpus[d]  # TODO: tune the loss weights and `flgs`
      reg_loss = ognn.loss.sdf_reg_loss(
          sdf, grads[d], batch['sdf'], batch['grad'], '_%d' % d)
      output.update(reg_loss)

    # vq loss
    flags = self.FLAGS.LOSS
    output['vq_loss'] = flags.vq_weight * model_out['vq_loss']
    return output

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    octree_in = batch['octree']
    octree_gt = OctreeD(batch['octree'])
    model_out = self.model(octree_in, octree_gt, batch['pos'])

    output = self.compute_loss(batch, model_out)
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

  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree'].cuda()
    octree_out = OctreeD(octree_in)
    output = self.model.forward(octree_in, octree_out, update_octree=True)

    # extract the mesh
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1:
      filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, filename + '.obj')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
      os.makedirs(folder)
    bbmin, bbmax = self._get_bbox(batch)
    utils.create_mesh(
        output['neural_mpu'], filename, size=self.FLAGS.SOLVER.resolution,
        bbmin=bbmin, bbmax=bbmax, mesh_scale=self.FLAGS.DATA.test.point_scale,
        save_sdf=self.FLAGS.SOLVER.save_sdf)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    points = batch['points_in'][0]
    points[:, :3] *= self.FLAGS.DATA.test.point_scale
    utils.points2ply(filename, batch['points_in'][0])

  def _init_octree_out(self, octree_in, depth_out):
    full_depth = octree_in.full_depth  # grow octree to full_depth
    octree_out = ocnn.octree.init_octree(
        depth_out, full_depth, octree_in.batch_size, octree_in.device)
    return OctreeD(octree_out, full_depth)

  def _get_bbox(self, batch):
    if 'bbox' in batch:
      bbox = batch['bbox'][0].numpy()
      bbmin, bbmax = bbox[:3], bbox[3:]
    else:
      sdf_scale = self.FLAGS.SOLVER.sdf_scale
      bbmin, bbmax = -sdf_scale, sdf_scale
    return bbmin, bbmax


if __name__ == '__main__':
  VAESolver.main()
