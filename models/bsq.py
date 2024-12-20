import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySphericalQuantizer(nn.Module):

  def __init__(self, embed_dim: int, gamma0: float = 1.0, gamma1: float = 1.0,
               inv_temperature: float = 1.0):
    super().__init__()
    self.embed_dim = embed_dim
    self.gamma0 = gamma0    # loss weight for entropy penalty
    self.gamma1 = gamma1    # loss weight for entropy penalty
    self.inv_temperature = inv_temperature
    self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))

  def quantize(self, z):
    assert z.shape[-1] == self.embed_dim
    zhat = (z > 0) * 2 - 1
    return z + (zhat - z).detach()

  def forward(self, z):
    z = F.normalize(z, p=2.0, dim=-1)

    persample_entropy, cb_entropy = self.soft_entropy_loss(z)
    entropy_penalty = self.gamma0 * persample_entropy - self.gamma1 * cb_entropy

    zq = self.quantize(z)
    indices = self.code2index(zq.detach())
    zq = zq * (1.0 / self.embed_dim ** 0.5)

    return zq, indices, entropy_penalty / self.inv_temperature,

  def soft_entropy_loss(self, z):
    r'''Compute the entropy loss for the soft quantization.'''

    p = torch.sigmoid(-4 * z / (self.embed_dim**0.5 * self.inv_temperature))
    prob = torch.stack([p, 1-p], dim=-1)
    per_sample_entropy = self.get_entropy(prob, dim=-1).sum(dim=-1).mean()

    # macro average of the probability of each subgroup
    avg_prob = torch.mean(prob, dim=0)
    codebook_entropy = self.get_entropy(avg_prob, dim=-1)

    # the approximation of the entropy is the sum of the entropy of each subgroup
    return per_sample_entropy, codebook_entropy.sum()

  def get_entropy(self, probs, dim=-1):
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
    return H

  def code2index(self, zhat):
    r'''Converts a `code` to an index in the codebook.
    Args:
        zhat: A tensor of shape (..., C) containing the codes. must be in {-1, 1}
    '''
    assert zhat.shape[-1] == self.embed_dim
    return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

  def index2code(self, indices):
    r'''Inverse of `indexes_to_codes`.'''
    indices = indices.unsqueeze(-1)
    binary_codes = torch.remainder(torch.floor_divide(indices, self.basis), 2)
    return binary_codes * 2 - 1

  def extract_code(self, indices):
    z_q = self.index2code(indices)
    z_q = z_q * (1. / self.embed_dim ** 0.5)
    return z_q
