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
from tqdm import tqdm


class OctarSolver(Solver):

    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = FLAGS.MODEL.depth
        self.depth_stop = FLAGS.MODEL.depth_stop
        self.full_depth = FLAGS.MODEL.full_depth
        self.enable_vqvae = FLAGS.MODEL.enable_vqvae

    def config_model(self):
        flags = self.FLAGS.MODEL
        model = GPT()
        vqvae = VQVAE(**flags.VQVAE)

        # load the pretrained vqvae
        checkpoint = torch.load(flags.vqvae_ckpt)
        vqvae.load_state_dict(checkpoint)
        print("Load VQVAE from", flags.vqvae_ckpt)

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
            self.model_module = model.module
            self.vqvae_module = vqvae.module
        else:
            self.model_module = model
            self.vqvae_module = vqvae
        self.model = model
        # if self.is_master:
        #     print(model)

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

        # For testing VQVAE
        # vq_code = self.vqvae.extract_code(octree_in)
        # zq, indices, _ = self.vqvae.quantizer(vq_code)
        # output = self.vqvae.decode_code(zq, self.depth_stop, OctreeD(octree_in), update_octree=True)
        # utils.create_mesh(output['neural_mpu'], os.path.join("mytools/0.obj"))

        split_seq = utils.octree2seq(
            octree_in, self.full_depth, self.depth_stop).long()
        output = self.model(split_seq, octree_in,
                            self.full_depth, self.depth_stop,
                            vqvae=self.vqvae_module if self.enable_vqvae else None)
        losses = [val for key, val in output.items() if 'loss' in key]
        output['loss'] = torch.sum(torch.stack(losses))
        return output

    def train_step(self, batch):
        output = self.model_forward(batch)
        output = {'train/' + key: val for key, val in output.items()}
        return output

    # rewrite the test_epoch function as generate
    def test_epoch(self, epoch):
        self.model.eval()
        self.generate_iter(0)

    def generate(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.load_checkpoint()
        for iter in tqdm(range(10000), ncols=80):
            self.generate_iter(iter)

    def generate_iter(self, iter):
        # forward the model
        octree_out = ocnn.octree.init_octree(
            self.depth, self.full_depth, 1, self.device)
        octree_out, vq_indices = self.model_module.generate(
            octree_out, depth_low=self.full_depth, depth_high=self.depth_stop,
            vqvae=self.vqvae_module if self.enable_vqvae else None)

        for d in range(self.full_depth + 1, self.depth_stop):
            utils.export_octree(octree_out, d, os.path.join(
                self.logdir, f'results/octree_depth{d}'))

        # decode the octree
        if self.enable_vqvae:
            zq = self.vqvae_module.quantizer.embedding(vq_indices)
            output = self.vqvae_module.decode_code(
                zq, self.depth_stop, OctreeD(octree_out), update_octree=True)

            # extract the mesh
            utils.create_mesh(
                output['neural_mpu'], os.path.join(self.logdir, f"results/{iter}.obj"), size=self.FLAGS.SOLVER.resolution,
                bbmin=-self.FLAGS.SOLVER.sdf_scale, bbmax=self.FLAGS.SOLVER.sdf_scale, mesh_scale=self.FLAGS.DATA.test.point_scale,
                save_sdf=self.FLAGS.SOLVER.save_sdf, with_color=self.FLAGS.SOLVER.with_color)

    def _init_octree_out(self, octree_in, depth_out):
        full_depth = octree_in.full_depth    # grow octree to full_depth
        octree_out = ocnn.octree.init_octree(
            depth_out, full_depth, octree_in.batch_size, octree_in.device)
        return OctreeD(octree_out, full_depth)


if __name__ == '__main__':
    OctarSolver.main()
