from models.vqvaev2 import VQVAE
from models.vae import VAE

def build_vqvae_model(flags):

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

  if flags.name.lower() == 'vqvae_small':
    return VQVAEs(**flags)  # Small VQVAE with 2.5M parameters
  elif flags.name.lower() == 'vqvae_big':
    return VQVAEb(**flags)
  elif flags.name.lower() == 'vqvae_large':
    return VQVAEl(**flags)
  elif flags.name.lower() == 'vae':
    return VAE(**flags)
  else:
    return VQVAE(**flags)   # Default VQVAE with 4.2M parameters.
