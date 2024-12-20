from einops import rearrange, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySphericalQuantizer(nn.Module):

  def __init__(self, embed_dim, gamma0, gamma, zeta,
               inv_temperature=1):
    super().__init__()
    self.embed_dim = embed_dim
    self.gamma0 = gamma0  # loss weight for entropy penalty
    self.gamma = gamma  # loss weight for entropy penalty
    self.zeta = zeta    # loss weight for entire entropy penalty
    self.inv_temperature = inv_temperature
    self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))

  def quantize(self, z):
    assert z.shape[-1] == self.embed_dim
    zhat = (z > 0) * 2 - 1
    return z + (zhat - z).detach()

  def forward(self, z):
    z = F.normalize(z, p=2.0, dim=-1)

    persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
    entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

    zq = self.quantize(z)
    indices = self.codes_to_indexes(zq.detach())
    zq = zq * (1.0 / self.embed_dim ** 0.5)

    return (zq, self.zeta * entropy_penalty / self.inv_temperature,
            {"indices": indices, "avg_prob": avg_prob})

  def soft_entropy_loss(self, z):
    r'''Compute the entropy loss for the soft quantization.'''

    p = torch.sigmoid(-4 * z / (self.embed_dim**0.5 * self.inv_temperature))
    prob = torch.stack([p, 1-p], dim=-1)
    per_sample_entropy = self.get_entropy(prob, dim=-1).sum(dim=-1).mean()

    # macro average of the probability of each subgroup
    avg_prob = reduce(prob, '... g d ->g d', 'mean')
    codebook_entropy = self.get_entropy(avg_prob, dim=-1)

    # the approximation of the entropy is the sum of the entropy of each subgroup
    return per_sample_entropy, codebook_entropy.sum(), avg_prob

  def get_entropy(self, probs, dim=-1):
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
    return H

  def codes_to_indexes(self, zhat):
    r'''Converts a `code` to an index in the codebook.
    Args:
        zhat: A tensor of shape (..., C) containing the codes. must be in {-1, 1}
    '''
    assert (zhat.shape[-1] == self.embed_dim,
            f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}")
    return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

  def indexes_to_codes(self, indices):
    r'''Inverse of `indexes_to_codes`.'''
    indices = indices.unsqueeze(-1)
    binary_codes = torch.remainder(torch.floor_divide(indices, self.basis), 2)
    return binary_codes * 2 - 1

  def get_codebook_entry(self, indices):
    z_q = self.indexes_to_codes(indices)
    q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
    z_q = z_q * q_scale
    if self.input_format == 'bchw':
      h, w = int(z_q.shape[1] ** 0.5)
      assert h * w == z_q.shape[1], 'Invalid sequence length'
      z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
    return z_q
