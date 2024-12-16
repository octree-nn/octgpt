import torch
import ocnn
import ognn
import torch.nn.functional as F

from typing import List, Optional
from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu
from models.vae import VectorQuantizer, VectorQuantizerG, VectorQuantizerN, VectorQuantizerP, VQVAE


class Encoder(torch.nn.Module):
  r''' An encoder takes an octree as input and outputs latent codes on a
  downsampled octree.
  '''

  def __init__(self, in_channels: int,
               channels: List[int] = [32, 32, 32, 64, 128, 256],
               resblk_nums: List[int] = [1, 1, 1, 2, 4, 2],
               code_stage_num: int = 4,
               bottleneck: int = 2, 
               **kwargs):
    super().__init__()
    groups = 32
    self.stage_num = len(channels)
    self.delta_depth = self.stage_num - 1
    self.code_stage_num = code_stage_num

    self.conv1 = ocnn.modules.OctreeConvGnRelu(in_channels, channels[0], groups)
    self.blocks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        channels[i], channels[i], resblk_nums[i], bottleneck, nempty=False,
        resblk=ocnn.modules.OctreeResBlockGn, use_checkpoint=True)
        for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvGnRelu(
        channels[i], channels[i+1], groups, kernel_size=[2], stride=2)
        for i in range(self.stage_num - 1)])  # Note: self.stage_num - 1

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = self.conv1(data, octree, depth)
    convs = {}
    for i in range(self.stage_num):
      di = depth - i
      out = self.blocks[i](out, octree, di)
      if self.stage_num - i <= self.code_stage_num:
        convs[di] = out
      if i < self.stage_num - 1:
        out = self.downsample[i](out, octree, di)
    return convs


class Decoder(torch.nn.Module):
  def __init__(self, n_node_type: int,
               channels: List[int] = [256, 128, 64, 32, 32, 32],
               blk_nums: List[int] = [2, 4, 2, 1, 1, 1],
               mpu_stage_nums: int = 6, pred_stage_nums: int = 3,
               bottleneck: int = 2, **kwargs):
    super().__init__()
    self.n_edge_type = 7
    self.head_channel = 64
    self.use_checkpoint = True
    self.act_type = 'relu'
    self.resblk_type = 'basic'
    self.norm_type = 'group_norm'
    self.n_node_type = n_node_type
    self.blk_nums = blk_nums
    self.channels = channels

    self.stages = len(self.blk_nums)
    self.graph_pad = ognn.nn.GraphPad()

    # tiny decoder
    n_node_type = self.n_node_type - self.stages + 1
    n_node_types = [n_node_type + i for i in range(self.stages)]
    self.upsample = torch.nn.ModuleList([ognn.nn.GraphUpsample(
        self.channels[i - 1], self.channels[i],
        self.norm_type, self.act_type) for i in range(1, self.stages)])
    self.decoder = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        self.channels[i], self.channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type,
        self.act_type, bottleneck, self.blk_nums[i],
        self.resblk_type) for i in range(self.stages)])

    # header
    self.start_pred = self.stages - pred_stage_nums
    self.start_mpu = self.stages - mpu_stage_nums
    # start_mpu <= start_pred
    # In training, start predicting signals from start_mpu for better code
    # In inference, start predicting signals from start_pred for better mesh
    self.predict = torch.nn.ModuleList([ognn.nn.Prediction(
        self.channels[i], self.head_channel, 2, self.norm_type,
        self.act_type) for i in range(self.start_mpu, self.stages)])
    self.regress = torch.nn.ModuleList([ognn.nn.Prediction(
        self.channels[i], self.head_channel, 4, self.norm_type,
        self.act_type) for i in range(self.start_mpu, self.stages)])

  def _octree_align(self, value: torch.Tensor, octree: OctreeD,
                    octree_query: OctreeD, depth: int):
    key = octree.graphs[depth].key
    query = octree_query.graphs[depth].key
    assert key.shape[0] == value.shape[0]
    return ocnn.nn.search_value(value, key, query)

  def octree_decoder(self, convs: dict, octree_in: OctreeD, octree_out: OctreeD,
                     depth: int, update_octree: bool = False):
    logits, signals = dict(), dict()
    deconv = octree_in.pad_zeros(convs[depth], depth)
    for i in range(self.stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octree_out, d-1)
        if d in convs:
          if i > self.start_pred:
            skip = self._octree_align(octree_in.pad_zeros(convs[d], d), octree_in, octree_out, d)
          else:
            skip = octree_in.pad_zeros(convs[d], d)
          deconv = deconv + skip  # skip connections
      deconv = self.decoder[i](deconv, octree_out, d)

      # predict the splitting label and signal
      if i >= self.start_mpu:
        j = i - self.start_mpu
        logit = self.predict[j](deconv, octree_out, d)
        nnum = octree_out.nnum[d]
        logits[d] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      if i >= self.start_mpu:
        j = i - self.start_mpu
        signal = self.regress[j](deconv, octree_out, d)
        signals[d] = self.graph_pad(signal, octree_out, d)

      # update the octree according to predicted labels
      if update_octree and i >= self.start_pred:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if i < self.stages - 1:
          octree_out.octree_grow(d + 1)

    return {'logits': logits, 'signals': signals, 'octree_out': octree_out}

  def forward(self, code: torch.Tensor, depth: int, octree_in: OctreeD,
              octree_out: OctreeD, pos: torch.Tensor = None,
              update_octree: bool = False):
    # run decoder
    output = self.octree_decoder(
        code, octree_in, octree_out, depth, update_octree)

    # setup mpu
    depth_out = octree_out.depth
    neural_mpu = mpu.NeuralMPU(output['signals'], octree_out, depth_out)
    if pos is not None:  # compute function value with mpu
      output['mpus'] = neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda p: neural_mpu(p)[depth_out]
    return output


class NVQVAE(torch.nn.Module):

  def __init__(self, in_channels: int,
               embedding_sizes: int = 128,
               embedding_channels: int = 64,
               feature: str = 'ND',
               n_node_type: int = 9,
               quantizer_type: str = 'plain',
               quantizer_group: int = 4,
               **kwargs):
    super().__init__()
    self.feature = feature
    self.config_network()

    self.encoder = Encoder(
        in_channels, self.enc_channels, self.enc_resblk_nums, self.code_stage_nums, self.bottleneck)
    self.decoder = Decoder(
        n_node_type, self.dec_channels, self.dec_resblk_nums, self.mpu_stage_nums, self.pred_stage_nums, self.bottleneck)
    self.quantizer = self.get_quantizer(
        quantizer_type, embedding_sizes, embedding_channels, quantizer_group)

    self.pre_proj = torch.nn.ModuleList([
        torch.nn.Linear(self.enc_channels[-(i + 1)], embedding_channels, bias=True)
        for i in range(self.code_stage_nums)])

    self.post_proj = torch.nn.ModuleList([
        torch.nn.Linear(embedding_channels, self.dec_channels[i], bias=True)
        for i in range(self.code_stage_nums)])

  def config_network(self):
    self.bottleneck = 2
    self.mpu_stage_nums = 6
    self.pred_stage_nums = 3
    self.code_stage_nums = 4

    self.enc_channels = [32, 32, 64, 128, 256, 512]
    self.enc_resblk_nums = [2, 2, 2, 4, 8, 2]

    self.dec_channels = [512, 256, 128, 64, 32, 32]
    self.dec_resblk_nums = [2, 8, 4, 2, 2, 2]

  def get_quantizer(self, quantizer_type: str, embedding_sizes: int,
                    embedding_channels: int, group: int = 4):
    kwargs = {'K': embedding_sizes, 'D': embedding_channels, 'G': group}

    if 'plain' in quantizer_type:
      Quantizer = VectorQuantizer
    elif 'project' in quantizer_type:
      Quantizer = VectorQuantizerP
    elif 'normalize' in quantizer_type:
      Quantizer = VectorQuantizerN
    else:
      raise NotImplementedError

    if 'group' in quantizer_type:
      kwargs['Q'] = Quantizer
      Quantizer = VectorQuantizerG

    return Quantizer(**kwargs)

  def forward(self, octree_in: Octree, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    code = self.extract_code(octree_in)
    zq, _, vq_loss = self.quantizer(code)
    octree_in = OctreeD(octree_in)
    code_depth = octree_in.depth - self.encoder.delta_depth + self.code_stage_nums - 1
    output = self.decode_code(zq, code_depth, octree_in, octree_out,
                              pos, update_octree)
    output['vae_loss'] = vq_loss
    return output

  def extract_code(self, octree_in: Octree):
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)
    start_depth = depth - self.encoder.delta_depth
    code = []
    for i in range(self.code_stage_nums):
      d = i + start_depth
      code.append(self.pre_proj[i](conv[d]))
    code = torch.cat(code, dim=0)
    return code

  def decode_code(self, code: torch.Tensor, code_depth: int, octree_in: OctreeD,
                  octree_out: OctreeD, pos: torch.Tensor = None,
                  update_octree: bool = False):
    # project the vae code to features
    data = {}
    cur_nnum = 0
    start_depth = code_depth - self.code_stage_nums + 1
    for i in range(self.code_stage_nums):
      d = i + start_depth
      nnum_d = octree_in.nnum[d]
      data[d] = self.post_proj[i](code[cur_nnum:cur_nnum+nnum_d])
      cur_nnum += nnum_d

    # run the decoder defined on dual octrees
    output = self.decoder(data, start_depth, octree_in, octree_out,
                          pos, update_octree)
    return output
