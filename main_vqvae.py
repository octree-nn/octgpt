import os
import torch
import ocnn
import ognn
from thsolver import Solver
from ognn.octreed import OctreeD

from utils import utils, builder
from datasets import get_shapenet_dataset


class VAESolver(Solver):

  def get_model(self, flags):
    return builder.build_vqvae_model(flags)

  def get_dataset(self, flags):
    return get_shapenet_dataset(flags)

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
    wg, ws, wm = 1.0, 200.0, 1.0
    flags = self.FLAGS.LOSS
    mpus = model_out['mpus']
    # R = [(d, w) for d, w in zip(flags.mpu_keys, flags.mpu_weights) if d in mpus]
    for d in mpus:
      sdf = mpus[d]
      grad = ognn.loss.compute_gradient(sdf, batch['pos'])[:, :3]
      grad_loss = (grad - batch['grad']).pow(2).mean() * (wg * wm)
      sdf_loss = (sdf - batch['sdf']).pow(2).mean() * (ws * wm)
      output['grad_loss_%d' % d] = grad_loss
      output['sdf_loss_%d' % d] = sdf_loss

    # vq loss
    output['vq_loss'] = flags.vq_weight * model_out['vq_loss']
    return output

  def model_forward(self, batch):
    self.batch_to_cuda(batch)
    octree_in = batch['octree_in']
    octree_gt = OctreeD(batch['octree_gt'])
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
        bbmin=bbmin, bbmax=bbmax, mesh_scale=flags.point_scale,
        save_sdf=flags.save_sdf)

    # save the input point cloud
    filename = filename[:-4] + '.input.ply'
    points = batch['points_in'][0]
    points.points *= flags.point_scale
    utils.points2ply(filename, points)

  def _extract_filename(self, batch):
    filename = batch['filename'][0]
    pos = filename.rfind('.')
    if pos != -1: filename = filename[:pos]  # remove the suffix
    filename = os.path.join(self.logdir, filename + '.obj')
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
