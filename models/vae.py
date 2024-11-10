import torch
import ocnn
import ognn

from typing import List
from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu
from models.vqvaev2 import VQVAE, Encoder, Decoder

class VAE(VQVAE):

  def __init__(self, in_channels: int,
               embedding_sizes: int = 128,
               embedding_channels: int = 64,
               feature: str = 'ND',
               n_node_type: int = 7, **kwargs):
    super().__init__(in_channels, embedding_sizes,
                     embedding_channels, feature, n_node_type)
    self.quantizer = None
    
    self.decoder = Decoder(
        n_node_type, self.dec_enc_channels, self.dec_enc_resblk_nums,
        self.dec_dec_channels, self.dec_dec_resblk_nums,
        mpu_stage_nums=6, pred_stage_nums=6)
    
    self.pre_proj = torch.nn.Linear(
        self.enc_channels[-1], embedding_channels * 2, bias=True)
    self.post_proj = torch.nn.Linear(
        embedding_channels, self.dec_enc_channels[0], bias=True)

  def forward(self, octree_in: Octree, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    code = self.extract_code(octree_in)
    posterior = DiagonalGaussianDistribution(code, kl_std = 0.25)
    z = posterior.sample()
    kl_loss = posterior.kl()
    code_depth = octree_in.depth - self.encoder.delta_depth
    octree_in = OctreeD(octree_in)
    output = self.decode_code(z, code_depth, octree_in, octree_out,
                              pos, update_octree)
    output['vq_loss'] = kl_loss.mean()
    return output



class DiagonalGaussianDistribution(object):
  def __init__(self, parameters, kl_std=1):
    self.parameters = parameters
    self.kl_std = kl_std
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self):
    x = self.mean + self.std * \
        torch.randn(self.mean.shape).to(device=self.parameters.device)
    return x

  def kl(self):
    if self.kl_std == 1:
      return 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
    else:
      gt_dist = torch.distributions.normal.Normal(
          torch.zeros_like(self.mean), torch.ones_like(self.std) * self.kl_std)
      sampled_dist = torch.distributions.normal.Normal(self.mean, self.std)
      kl = torch.distributions.kl.kl_divergence(
          sampled_dist, gt_dist)  # reversed KL
      return kl