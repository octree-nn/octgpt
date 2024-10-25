import torch
import ocnn
import ognn

from typing import List
from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu


class Encoder(torch.nn.Module):
  r''' An encoder takes an octree as input and outputs latent codes
  on a downsampled octree.
  '''

  def __init__(self, in_channels: int,
               channels: List[int] = [32, 64],
               resblk_nums: List[int] = [1, 1],
               groups: int = 32, **kwargs):
    super().__init__()
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
    # self.project = torch.nn.Linear(channels[-1], h_channels, use_bias=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = self.conv1(data, octree, depth)
    for i in range(self.stage_num):
      di = depth - i
      out = self.blocks[i](out, octree, di)
      if i < self.stage_num - 1:
        out = self.downsample[i](out, octree, di)
    # out = self.project(out)
    return out


class Decoder(torch.nn.Module):
  r'''
  '''

  def __init__(self, out_channels: int,
               n_node_type: int = 7,
               channels: List[int] = [64, 32],
               resblk_nums: List[int] = [1, 1],
               unet_channels: List[int] = [64, 128, 256],
               unet_resblk_nums: List[int] = [2, 2, 2],
               predict_octree: bool = False, **kwargs):
    super().__init__()
    self.delta_depth = len(channels) - 1
    n_node_type = n_node_type - self.delta_depth
    self.unet = TinyUNet(
        unet_channels, unet_resblk_nums, resblk_type='basic',
        bottleneck=2, n_node_type=n_node_type)
    self.decoding = DecodeOctree(
        channels, resblk_nums, resblk_type='basic',
        bottleneck=4, n_node_type=n_node_type, out_channels=out_channels,
        predict_octree=predict_octree)

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    data = self.unet(data, octree, depth)
    output = self.decoding(data, octree, depth, update_octree)
    return output


class TinyUNet(torch.nn.Module):
  r''' A tiny U-Net used internally in the decoder to increase the
  ability of the decoder, which takes dual octrees as input and output.
  '''

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, **kwargs):
    super().__init__()
    n_edge_type = 7
    act_type = 'relu'
    norm_type = 'group_norm'
    use_checkpoint = True
    self.stage_num = len(channels)
    self.delta_depth = len(channels) - 1

    n_node_types = [n_node_type - i for i in range(self.stage_num)]
    self.encoder_blocks = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([ognn.nn.GraphDownsample(
        channels[i], channels[i+1], norm_type, act_type)
        for i in range(self.stage_num - 1)])    # Note: stage_num - 1

    self.decoder_blocks = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num - 1, -1, -1)])
    self.upsample = torch.nn.ModuleList([ognn.nn.GraphUpsample(
        channels[i], channels[i-1], norm_type, act_type)
        for i in range(self.stage_num - 1,  0, -1)])

  def encoder(self, data: torch.Tensor, octree: OctreeD, depth: int):
    out = dict()
    out[depth] = data
    for i in range(self.stage_num):
      di = depth - i
      out[di] = self.encoder_blocks[i](out[di], octree, di)
      if i < self.stage_num - 1:
        out[di-1] = self.downsample[i](out[di], octree, di)
    return out

  def decoder(self, datas: torch.Tensor, octree: OctreeD, depth: int):
    out = datas[depth]
    for i in range(self.stage_num):
      di = depth + i
      out = self.decoder_blocks[i](out, octree, di)
      if i < self.stage_num - 1:
        out = self.upsample[i](out, octree, di)
        out = out + datas[di+1]  # skip connections
    return out

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int):
    encs = self.encoder(data, octree, depth)
    out = self.decoder(encs, octree, depth - self.delta_depth)
    return out


class DecodeOctree(torch.nn.Module):

  def __init__(self, channels: List[int], resblk_nums: List[int],
               resblk_type: str = 'basic', bottleneck: int = 2,
               n_node_type: int = -1, out_channels: int = 4,
               predict_octree: bool = False, **kwargs):
    super().__init__()
    n_edge_type = 7
    act_type = 'gelu'
    norm_type = 'group_norm'
    use_checkpoint = True
    mid_channels = 32
    self.stage_num = len(channels)
    self.predict_octree = predict_octree

    n_node_types = [n_node_type + i for i in range(self.stage_num)]
    self.blocks = torch.nn.ModuleList([ognn.nn.GraphResBlocks(
        channels[i], channels[i], n_edge_type, n_node_types[i], norm_type,
        act_type, bottleneck, resblk_nums[i], resblk_type, use_checkpoint)
        for i in range(self.stage_num)])
    self.upsample = torch.nn.ModuleList([ognn.nn.GraphUpsample(
        channels[i], channels[i+1], norm_type, act_type)
        for i in range(self.stage_num - 1)])
    self.graph_pad = ognn.nn.GraphPad()

    self.regress = torch.nn.ModuleList([ognn.nn.Prediction(
        channels[i], mid_channels, out_channels, norm_type, act_type)
        for i in range(self.stage_num)])
    if predict_octree:
      self.predict = torch.nn.ModuleList([ognn.nn.Prediction(
          channels[i], mid_channels, 2, norm_type, act_type)
          for i in range(self.stage_num)])

  def forward(self, data: torch.Tensor, octree: OctreeD, depth: int,
              update_octree: bool = False):
    logits, signals = dict(), dict()
    for i in range(self.stage_num):
      di = depth + i
      data = self.blocks[i](data, octree, di)

      # predict the splitting label and signal
      if self.predict_octree:
        logit = self.predict[i](data, octree, di)
        nnum = octree.nnum[di]
        logits[di] = logit[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      signal = self.regress[i](data, octree, di)
      signals[di] = self.graph_pad(signal, octree, di)

      # update the octree according to predicted labels
      if update_octree and self.predict_octree:
        split = logits[di].argmax(1).int()
        octree.octree_split(split, di)
        if i < self.stage_num - 1:
          octree.octree_grow(di + 1)

      # upsample
      if i < self.stage_num - 1:
        data = self.upsample[i](data, octree, di)

    return {'logits': logits, 'signals': signals, 'octree_out': octree}


class VQVAE(torch.nn.Module):

  def __init__(self, in_channels: int,
               out_channels: int = 4,
               embedding_sizes: int = 128,
               embedding_channels: int = 256,
               feature: str = 'ND',
               groups: int = 32,
               n_node_type: int = 7, **kwargs):
    super().__init__()
    self.feature = feature
    self.config_network()

    self.encoder = Encoder(
        in_channels, self.enc_channels, self.enc_resblk_nums, groups)
    self.decoder = Decoder(
        out_channels, n_node_type, self.dec_channels, self.dec_resblk_nums,
        self.dec_net_channels, self.dec_net_resblk_nums, predict_octree=True)

    self.pre_proj = torch.nn.Linear(
        self.enc_channels[-1], embedding_channels, bias=True)
    self.post_proj = torch.nn.Linear(
        embedding_channels, self.dec_channels[0], bias=True)

  def config_network(self):
    self.enc_channels = [32, 32, 64]
    self.enc_resblk_nums = [1, 1, 1]

    self.dec_channels = [64, 32, 32]
    self.dec_resblk_nums = [1, 1, 1]
    self.dec_net_channels = [64, 128, 256]
    self.dec_net_resblk_nums = [1, 1, 1]

  def forward(self, octree_in: Octree, octree_out: OctreeD,
              pos: torch.Tensor = None, update_octree: bool = False):
    code = self.extract_code(octree_in)

    # TODO: Add vqvae loss here

    code_depth = octree_in.depth - self.encoder.delta_depth
    output = self.decode_code(code, code_depth, octree_out, pos, update_octree)

    # output['vq_loss'] = posterior.kl().mean()
    # output['code_max'] = z.max()
    # output['code_min'] = z.min()
    return output

  def extract_code(self, octree_in: Octree):
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)
    code = self.pre_proj(conv)    # project features to the vae code
    return code

  def decode_code(self, code: torch.Tensor, code_depth: int, octree: OctreeD,
                  pos: torch.Tensor = None, update_octree: bool = False):
    # project the vae code to features
    data = self.post_proj(code)

    # `data` is defined on the octree, here we need pad zeros to be compatible
    # with the dual octree
    data = octree.pad_zeros(data, code_depth)

    # run the decoder defined on dual octrees
    output = self.decoder(data, octree, code_depth, update_octree)

    # setup mpu
    depth_out = octree.depth
    neural_mpu = mpu.NeuralMPU(output['signals'], octree, depth_out)

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = neural_mpu(pos)

    # create the mpu wrapper
    output['neural_mpu'] = lambda pos: neural_mpu(pos)[depth_out]
    return output
