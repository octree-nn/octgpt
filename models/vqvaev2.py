import torch
import ocnn
import ognn

from typing import List
from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu


class Encoder(torch.nn.Module):
  r''' An encoder takes an octree as input and outputs latent codes on a
  downsampled octree.
  '''

  def __init__(self, in_channels: int,
               channels: List[int] = [32, 32, 64],
               resblk_nums: List[int] = [1, 1, 1], **kwargs):
    super().__init__()
    groups = 32
    self.stage_num = len(channels)
    self.delta_depth = self.stage_num - 1

    self.conv1 = ocnn.modules.OctreeConvGnRelu(in_channels, channels[0], groups)
    self.blocks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        channels[i], channels[i], resblk_nums[i], bottleneck=2, nempty=False,
        resblk=ocnn.modules.OctreeResBlockGn, use_checkpoint=True)
        for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvGnRelu(
        channels[i], channels[i+1], groups, kernel_size=[2], stride=2)
        for i in range(self.stage_num - 1)])  # Note: self.stage_num - 1

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = self.conv1(data, octree, depth)
    for i in range(self.stage_num):
      di = depth - i
      out = self.blocks[i](out, octree, di)
      if i < self.stage_num - 1:
        out = self.downsample[i](out, octree, di)
    return out


class Decoder(torch.nn.Module):
  r''' A decoder takes a downsampled octree and latent codes as input and
  outputs the upsampled octree. This decoder is designed to take dual octrees
  as input and output. The output octree is converted to a continuous surface
  via MPU. It contains a tiny U-Net to increase the ability of the decoder.
  '''

  def __init__(self, n_node_type: int,
               encoder_channels: List[int] = [32, 64, 128, 256],
               encoder_blk_nums: List[int] = [1, 2, 4, 2],
               decoder_channels: List[int] = [256, 128, 64, 32, 32, 32],
               decoder_blk_nums: List[int] = [2, 4, 2, 1, 1, 1],
               mpu_stage_nums: int = 2,
               **kwargs):
    super().__init__()
    self.bottleneck = 2
    self.n_edge_type = 7
    self.head_channel = 64
    self.use_checkpoint = True
    self.act_type = 'relu'
    self.resblk_type = 'basic'
    self.norm_type = 'group_norm'
    self.n_node_type = n_node_type
    self.encoder_blk_nums = encoder_blk_nums
    self.decoder_blk_nums = decoder_blk_nums
    self.encoder_channels = encoder_channels
    self.decoder_channels = decoder_channels

    self.encoder_stages = len(self.encoder_blk_nums)
    self.decoder_stages = len(self.decoder_blk_nums)
    self.graph_pad = ognn.nn.GraphPad()

    # tiny encoder
    n_node_types = [self.n_node_type - i for i in range(self.encoder_stages)]
    self.encoder = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        self.encoder_channels[i], self.encoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type,
        self.act_type, self.bottleneck, self.encoder_blk_nums[i],
        self.resblk_type) for i in range(self.encoder_stages)])
    self.downsample = torch.nn.ModuleList([ognn.nn.GraphDownsample(
        self.encoder_channels[i], self.encoder_channels[i+1],
        self.norm_type, self.act_type) for i in range(self.encoder_stages - 1)])

    # tiny decoder
    n_node_type = self.n_node_type - self.encoder_stages + 1
    n_node_types = [n_node_type + i for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList([ognn.nn.GraphUpsample(
        self.decoder_channels[i - 1], self.decoder_channels[i],
        self.norm_type, self.act_type) for i in range(1, self.decoder_stages)])
    self.decoder = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        self.decoder_channels[i], self.decoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type,
        self.act_type, self.bottleneck, self.decoder_blk_nums[i],
        self.resblk_type) for i in range(self.decoder_stages)])

    # header
    self.predict = torch.nn.ModuleList([ognn.nn.Prediction(
        self.decoder_channels[i], self.head_channel, 2, self.norm_type,
        self.act_type) for i in range(self.decoder_stages)])
    self.start_mpu = self.decoder_stages - mpu_stage_nums
    self.regress = torch.nn.ModuleList([ognn.nn.Prediction(
        self.decoder_channels[i], self.head_channel, 4, self.norm_type,
        self.act_type) for i in range(self.start_mpu, self.decoder_stages)])

  def _octree_align(self, value: torch.Tensor, octree: OctreeD,
                    octree_query: OctreeD, depth: int):
    key = octree.graphs[depth].key
    query = octree_query.graphs[depth].key
    assert key.shape[0] == value.shape[0]
    return ocnn.nn.search_value(value, key, query)

  def octree_encoder(self, code: torch.Tensor, octree: OctreeD, depth: int):
    convs = {depth: code}  # initialize `convs` to save convolution features
    for i in range(self.encoder_stages):
      d = depth - i
      convs[d] = self.encoder[i](convs[d], octree, d)
      if i < self.encoder_stages - 1:
        convs[d-1] = self.downsample[i](convs[d], octree, d)
    return convs

  def octree_decoder(self, convs: dict, octree_in: OctreeD, octree_out: OctreeD,
                     depth: int, update_octree: bool = False):
    logits, signals = dict(), dict()
    deconv = convs[depth]
    for i in range(self.decoder_stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        if d in convs:
          skip = self._octree_align(convs[d], octree_in, octree_out, d)
          deconv = deconv + skip  # output-guided skip connections
      deconv = self.decoder[i](deconv, octree_out, d)

      # predict the splitting label and signal
      logit = self.predict[i](deconv, octree_out, d)
      nnum = octree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      if i >= self.start_mpu:
        j = i - self.start_mpu
        signal = self.regress[j](deconv, octree_out, d)
        signals[d] = self.graph_pad(signal, octree_out, d)

      # update the octree according to predicted labels
      if update_octree:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if i < self.decoder_stages - 1:
          octree_out.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree_out}

  def forward(self, code: torch.Tensor, depth: int, octree_in: OctreeD,
              octree_out: OctreeD, pos: torch.Tensor = None,
              update_octree: bool = False):
    # run encoder and decoder
    convs = self.octree_encoder(code, octree_in, depth)
    depth = depth - self.encoder_stages + 1
    output = self.octree_decoder(convs, octree_in, octree_out,
                                 depth, update_octree)

    # setup mpu
    depth_out = octree_out.depth
    neural_mpu = mpu.NeuralMPU(output['signals'], octree_out, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: neural_mpu(pos)[depth_out]
    return output


class VQVAE(torch.nn.Module):

  def __init__(self, in_channels: int,
               embedding_sizes: int = 128,
               embedding_channels: int = 64,
               feature: str = 'ND',
               n_node_type: int = 7, **kwargs):
    super().__init__()
    self.feature = feature
    self.config_network()

    self.encoder = Encoder(
        in_channels, self.enc_channels, self.enc_resblk_nums)
    self.decoder = Decoder(
        n_node_type, self.dec_enc_channels, self.dec_enc_resblk_nums,
        self.dec_dec_channels, self.dec_dec_resblk_nums)
    self.quantizer = VectorQuantizer(embedding_sizes, embedding_channels)

    self.pre_proj = torch.nn.Linear(
        self.enc_channels[-1], embedding_channels, bias=True)
    self.post_proj = torch.nn.Linear(
        embedding_channels, self.dec_enc_channels[0], bias=True)

  def config_network(self):
    self.enc_channels = [32, 32, 64]
    self.enc_resblk_nums = [1, 1, 1]

    self.dec_enc_channels = [32, 64, 128, 256]
    self.dec_enc_resblk_nums = [1, 2, 4, 2]
    self.dec_dec_channels = [256, 128, 64, 32, 32, 32]
    self.dec_dec_resblk_nums = [2, 4, 2, 2, 1, 1]

  def forward(self, octree_in: Octree, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    code = self.extract_code(octree_in)
    zq, _, vq_loss = self.quantizer(code)
    code_depth = octree_in.depth - self.encoder.delta_depth
    octree_in = OctreeD(octree_in)
    output = self.decode_code(zq, code_depth, octree_in, octree_out,
                              pos, update_octree)
    output['vq_loss'] = vq_loss
    return output

  def extract_code(self, octree_in: Octree):
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)
    code = self.pre_proj(conv)    # project features to the vae code
    return code

  def decode_code(self, code: torch.Tensor, code_depth: int, octree_in: OctreeD,
                  octree_out: OctreeD, pos: torch.Tensor = None,
                  update_octree: bool = False):
    # project the vae code to features
    data = self.post_proj(code)

    # `data` is defined on the octree, here we need pad zeros to be compatible
    # with the dual octree
    data = octree_in.pad_zeros(data, code_depth)

    # run the decoder defined on dual octrees
    output = self.decoder(data, code_depth, octree_in, octree_out,
                          pos, update_octree)

    # setup mpu
    depth_out = octree_out.depth
    neural_mpu = mpu.NeuralMPU(output['signals'], octree_out, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: neural_mpu(pos)[depth_out]
    return output


class VectorQuantizer(torch.nn.Module):

  def __init__(self, K: int, D: int, beta: float = 0.5):
    super().__init__()
    self.beta = beta
    self.embedding = torch.nn.Embedding(K, D)
    self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

  def get_embedding_indices(self, z: torch.Tensor):
    # distances from z to embeddings e,
    # z: (N, D), e: (K, D)
    # (z - e)^2 = z^2 + e^2 - 2 e * z
    d = (torch.sum(z**2, dim=1, keepdim=True) +
         torch.sum(self.embedding.weight**2, dim=1) -
         2 * torch.matmul(z, self.embedding.weight.T))
    out = torch.argmin(d, dim=1)
    return out

  def forward(self, z):
    # get the closest embedding indices
    indices = self.get_embedding_indices(z)

    # get the embeddings
    zq = self.embedding(indices)

    # compute loss for the embedding
    loss = (self.beta * torch.mean((zq.detach() - z)**2) +
            torch.mean((zq - z.detach())**2))

    # preserve gradients: Straight-Through gradients
    zq = z + (zq - z).detach()
    return zq, indices, loss


class VectorQuantizerG(torch.nn.Module):

  def __init__(self, K: int, D: int, beta: float = 0.5, G: int = 4):
    super().__init__()
    C = D // G
    self.groups = G
    self.channels_per_group = C
    self.quantizers = torch.nn.ModuleList([
        VectorQuantizer(K, C, beta) for _ in range(G)])

  def forward(self, z):
    zqs = [None] * self.groups
    losses = [None] * self.groups
    z = z.view(-1, self.groups, self.channels_per_group)
    for i in range(self.groups):
      zqs[i], losses[i] = self.quantizers[i](z[:, i])
    zq = torch.cat(zqs, dim=1)
    loss = torch.mean(torch.stack(losses))
    return zq, loss
