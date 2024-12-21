import os
import torch
import ocnn
import ognn
import torch.nn.functional as F
from thsolver import Solver
from ognn.octreed import OctreeD

from utils import utils, builder
from datasets import get_shapenet_dataset


class VAESolver(Solver):

  def get_model(self, flags):
    return builder.build_vae_model(flags)

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
    output = dict()
    wo = [1.0] * 8 + [0.1] * 3  # lower weights for deep layers
    logits = model_out['logits']
    for d in logits.keys():
      label_gt = model_out['octree_out'].nempty_mask(d).long()
      output['loss_%d' % d] = F.cross_entropy(logits[d], label_gt) * wo[d]
      output['accu_%d' % d] = logits[d].argmax(1).eq(label_gt).float().mean()

    # regression loss
    wg, ws, wm = 1.0, 200.0, 1.0
    flags = self.FLAGS.LOSS
    mpus = model_out['mpus']
    for d in mpus:
      sdf = mpus[d]
      grad = ognn.loss.compute_gradient(sdf, batch['pos'])[:, :3]
      grad_loss = (grad - batch['grad']).pow(2).mean() * (wg * wm)
      sdf_loss = (sdf - batch['sdf']).pow(2).mean() * (ws * wm)
      output['grad_loss_%d' % d] = grad_loss
      output['sdf_loss_%d' % d] = sdf_loss

    # vae loss
    output['vae_loss'] = flags.vae_weight * model_out['vae_loss']
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
  
  def test_epoch(self, epoch):
    torch.cuda.empty_cache()
    # set test_every_epoch to 1, so that we can save checkpoints every epoch
    # Test the model every 5 epochs
    if epoch % 5 == 0:
      super().test_epoch(epoch)

  def eval_step(self, batch):
    
    # For data with 5 categories, evaluate random sampled data for time saving
    # import random
    # x = random.random()
    # if x > 0.2:
    #   return 
    
    # forward the model
    octree_in = batch['octree_in'].cuda()
    octree_out = OctreeD(octree_in)  # initialize
    # octree_out = self._init_octree_out(octree_in)
    import copy
    for d in range(6, 9):
      octree_out = copy.deepcopy(octree_in)
      octree_out.depth = d
      octree_out = OctreeD(octree_out)
      self.model.decoder.stages = d - octree_in.full_depth + 1
      
      # useless
      # if d <= 6:
      #   self.model.decoder.start_pred = d - octree_in.full_depth
      
      output = self.model(octree_in, octree_out, update_octree=True)

      # extract the mesh
      flags = self.FLAGS.DATA.test
      filename = self._extract_filename(batch)
      filename = filename.replace(self.logdir, os.path.join(self.logdir, f"depth_{d}"))
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
