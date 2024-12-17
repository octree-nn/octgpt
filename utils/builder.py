from models.vae import VQVAE, VAE

def build_vae_model(flags):

  class VQVAEs(VQVAE):
    def config_network(self):
      self.bottleneck = 2
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 3

      self.enc_channels = [32, 32, 64]
      self.enc_resblk_nums = [1, 1, 1]

      self.dec_enc_channels = [64, 128, 256]
      self.dec_enc_resblk_nums = [1, 1, 1]
      self.dec_dec_channels = [256, 128, 64, 32, 32]
      self.dec_dec_resblk_nums = [1, 1, 1, 2, 1]

  class VQVAEb(VQVAE):
    def config_network(self):
      self.bottleneck = 2
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 3

      self.enc_channels = [32, 32, 64]
      self.enc_resblk_nums = [2, 2, 2]

      self.dec_enc_channels = [32, 64, 128, 256]
      self.dec_enc_resblk_nums = [2, 4, 4, 2]
      self.dec_dec_channels = [256, 128, 64, 32, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 4, 2, 2, 2]

  class VQVAEl(VQVAEb):
    def config_network(self):
      super().config_network()
      self.bottleneck = 1

  class VQVAEh(VQVAE):
    def config_network(self):
      self.bottleneck = 1
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 3

      self.enc_channels = [32, 32, 64]
      self.enc_resblk_nums = [2, 2, 2]

      self.dec_enc_channels = [64, 128, 256, 512]
      self.dec_enc_resblk_nums = [2, 4, 8, 2]
      self.dec_dec_channels = [512, 256, 128, 64, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 8, 2, 2, 2]
 
  class VQVAEd5l(VQVAE):
    def config_network(self):
      self.bottleneck = 1
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 4

      self.enc_channels = [32, 32, 64, 64]
      self.enc_resblk_nums = [2, 2, 2, 2]

      self.dec_enc_channels = [128, 256, 256]
      self.dec_enc_resblk_nums = [4, 8, 2]
      self.dec_dec_channels = [256, 256, 128, 64, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 8, 2, 2, 2]
  
  class VQVAEd5h(VQVAE):
    def config_network(self):
      self.bottleneck = 1
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 4

      self.enc_channels = [32, 32, 64, 64]
      self.enc_resblk_nums = [2, 2, 2, 2]

      self.dec_enc_channels = [128, 256, 512]
      self.dec_enc_resblk_nums = [4, 8, 2]
      self.dec_dec_channels = [512, 256, 128, 64, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 8, 2, 2, 2]
 
  class VAEd5(VAE):
    def config_network(self):
      self.bottleneck = 2
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 4

      self.enc_channels = [32, 32, 64, 64]
      self.enc_resblk_nums = [1, 1, 1, 2]

      self.dec_enc_channels = [64, 128, 256]
      self.dec_enc_resblk_nums = [2, 4, 2]
      self.dec_dec_channels = [256, 128, 64, 32, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 2, 2, 1, 1]
 
  class VQVAEd5(VQVAE):
    def config_network(self):
      self.bottleneck = 2
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 4

      self.enc_channels = [32, 32, 64, 64]
      self.enc_resblk_nums = [1, 1, 1, 2]

      self.dec_enc_channels = [64, 128, 256]
      self.dec_enc_resblk_nums = [2, 4, 2]
      self.dec_dec_channels = [256, 128, 64, 32, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 2, 2, 1, 1]
 
  if flags.name.lower() == 'vqvae_small':
    return VQVAEs(**flags)  # Small VQVAE with 2.5M parameters
  elif flags.name.lower() == 'vqvae_big':
    return VQVAEb(**flags)  # Big VQVAE with 4.6M parameters
  elif flags.name.lower() == 'vqvae_large':
    return VQVAEl(**flags)  # Large VQVAE with 8.0M parameters
  elif flags.name.lower() == 'vqvae_huge':
    return VQVAEh(**flags)  # Large VQVAE with 33.9M parameters
  elif flags.name.lower() == 'vae':
    return VAE(**flags)     # Default VAE with 4.2M parameters
  elif flags.name.lower() == 'vae_d5':
    return VAEd5(**flags)   # Default depth-5 VAE with 4.52M parameters
  elif flags.name.lower() == 'vqvae_d5':
    return VQVAEd5(**flags) # Default depth-5 VQVAE with 4.54M parameters
  elif flags.name.lower() == 'vqvae_d5_large':
    model = VQVAEd5l(**flags)
    return model          # Large depth-5 VQVAE with 21.4M parameters
  elif flags.name.lower() == 'vqvae_d5_huge':
    model = VQVAEd5h(**flags)
    return model          # Huge depth-5 VQVAE with 34.3M parameters
  else:
    return VQVAE(**flags)   # Default VQVAE with 4.2M parameters
