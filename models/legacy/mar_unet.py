class MARUNet(MAR):
  def _init_blocks(self):
    self.blocks = OctFormer(
        channels=self.num_embed, num_blocks=self.num_blocks, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.ln_x = nn.LayerNorm(self.num_embed)

    self.vq_encoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_vq_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.downsample = ocnn.modules.OctreeConvGnRelu(
        self.num_embed, self.num_embed, group=32, kernel_size=[2], stride=2)

    self.vq_decoder = OctFormer(
        channels=self.num_embed, num_blocks=self.num_vq_blocks//2, num_heads=self.num_heads,
        patch_size=self.patch_size, dilation=self.dilation,
        attn_drop=self.drop_rate, proj_drop=self.drop_rate,
        pos_emb=eval(self.pos_emb_type), nempty=False,
        use_checkpoint=self.use_checkpoint, use_swin=self.use_swin)
    self.upsample = ocnn.modules.OctreeDeconvGnRelu(
        self.num_embed, self.num_embed, group=32, kernel_size=[2], stride=2)

  def forward_model(self, x, octree, depth_low, depth_high, mask, nnum_split):
    # if only split signals, not use vq sample
    apply_vq_sample = nnum_split < x.shape[0]
    if apply_vq_sample:
      depth_list_main = list(range(depth_low, depth_high)) + [depth_high - 1]
    else:
      depth_list_main = list(range(depth_low, depth_high + 1))

    depth_list_vq = list(range(depth_low, depth_high + 1))
    octreeT_vq = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation, nempty=False,
        depth_list=depth_list_vq, use_swin=self.use_swin, use_flex=self.use_flex)
    x = self.forward_blocks(x, octreeT_vq, self.vq_encoder)
    if apply_vq_sample:
      x_split = x[:nnum_split]
      x_vq = x[nnum_split:]
      x_vq = self.downsample(x_vq, octree, depth_high)
      x = torch.cat([x_split, x_vq], dim=0)

    octreeT = OctreeT(
        octree, x.shape[0], self.patch_size, self.dilation,
        nempty=False, depth_list=depth_list_main, use_swin=self.use_swin)
    x = self.forward_blocks(
        x, octreeT, self.blocks)

    if apply_vq_sample:
      # skip connection
      x_split = x[:nnum_split]
      x_vq = x[nnum_split:] + x_vq
      x_vq = self.upsample(x_vq, octree, depth_high - 1)
      x = torch.cat([x_split, x_vq], dim=0)
    x = self.forward_blocks(x, octreeT_vq, self.vq_decoder)

    x = self.ln_x(x)
    return x
