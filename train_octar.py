import os
import torch
import ocnn
import ognn
from thsolver import Solver
from ognn.octreed import OctreeD

import utils
from models.vqvae import VQVAE
from models.gpt import GPT
from datasets import get_shapenet_vae_dataset


class OctarSolver(Solver):
    
    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = FLAGS.MODEL.depth
        self.depth_stop = FLAGS.MODEL.depth_stop
        self.full_depth = FLAGS.MODEL.full_depth
    
    def config_model(self):
        flags = self.FLAGS.MODEL
        model = GPT()
        vqvae = VQVAE(**flags.VQVAE)
        model.cuda(device=self.device)
        vqvae.cuda(device=self.device)
        if self.world_size > 1:
            if flags.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                vqvae = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vqvae)
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[self.device],
                output_device=self.device, broadcast_buffers=False,
                find_unused_parameters=flags.find_unused_parameters)
            vqvae = torch.nn.parallel.DistributedDataParallel(
                module=vqvae, device_ids=[self.device],
                output_device=self.device, broadcast_buffers=False,
                find_unused_parameters=flags.find_unused_parameters)
        if self.is_master:
            print(model)
        self.model = model
        self.vqvae = vqvae

    def get_dataset(self, flags):
        return get_shapenet_vae_dataset(flags)

    def batch_to_cuda(self, batch):
        keys = ['octree', 'octree_in', 'octree_gt', 'pos', 'sdf',
                'grad', 'weight', 'occu', 'color']
        for key in keys:
            if key in batch:
                batch[key] = batch[key].cuda()

    def model_forward(self, batch):
        self.batch_to_cuda(batch)
        octree_in = batch['octree']
        
        split_seq = utils.octree2seq(octree_in, self.full_depth, self.depth_stop)
        loss = self.model(split_seq, octree_in, self.full_depth, self.depth_stop)

        output = {'loss': loss}
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
            filename = filename[:pos]    # remove the suffix
        filename = os.path.join(self.logdir, filename + '.obj')
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        bbmin, bbmax = self._get_bbox(batch)
        utils.create_mesh(
            output['neural_mpu'], filename, size=self.FLAGS.SOLVER.resolution,
            bbmin=bbmin, bbmax=bbmax, mesh_scale=self.FLAGS.DATA.test.point_scale,
            save_sdf=self.FLAGS.SOLVER.save_sdf, with_color=self.FLAGS.SOLVER.with_color)

        # save the input point cloud
        filename = filename[:-4] + '.input.ply'
        points = batch['points_in'][0]
        points[:, :3] *= self.FLAGS.DATA.test.point_scale
        utils.points2ply(filename, batch['points_in'][0])

    def _init_octree_out(self, octree_in, depth_out):
        full_depth = octree_in.full_depth    # grow octree to full_depth
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
    OctarSolver.main()
