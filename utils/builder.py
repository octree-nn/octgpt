from models.vae import VQVAE, VAE
from datasets import get_shapenet_dataset, get_synthetic_room_dataset


def build_vae_model(flags):
  class VQVAEb(VQVAE):
    def config_network(self):
      self.bottleneck = 1
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 3

      self.enc_channels = [32, 32, 64]
      self.enc_resblk_nums = [2, 2, 2]

      self.dec_enc_channels = [32, 64, 128, 256]
      self.dec_enc_resblk_nums = [2, 4, 4, 2]
      self.dec_dec_channels = [256, 128, 64, 32, 32, 32]
      self.dec_dec_resblk_nums = [2, 4, 4, 2, 2, 2]

  class VQVAEl(VQVAE):
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
  
  class VQVAEh(VQVAE):
    def config_network(self):
      self.bottleneck = 1
      self.mpu_stage_nums = 3
      self.pred_stage_nums = 3

      self.enc_channels = [32, 64, 128]
      self.enc_resblk_nums = [2, 2, 2]

      self.dec_enc_channels = [128, 256, 512, 1024]
      self.dec_enc_resblk_nums = [2, 4, 4, 4]
      self.dec_dec_channels = [1024, 512, 256, 128, 64, 32]
      self.dec_dec_resblk_nums = [4, 4, 4, 2, 2, 2]

  if flags.name.lower() == 'vqvae_big':
    return VQVAEb(**flags)  # Big VQVAE with 8.0M parameters
  elif flags.name.lower() == 'vqvae_large':
    return VQVAEl(**flags)  # Large VQVAE with 33.9M parameters
  elif flags.name.lower() == 'vqvae_huge':
    return VQVAEh(**flags)  # Large VQVAE with 76.3M parameters
  elif flags.name.lower() == 'vae':
    return VAE(**flags)     # Default VAE with 4.2M parameters
  else:
    return VQVAE(**flags)   # Default VQVAE with 4.2M parameters


def build_dataset(flags):
  if flags.name == 'shapenet' or flags.name == 'objaverse':
    return get_shapenet_dataset(flags)
  elif flags.name == 'synthetic_room':
    return get_synthetic_room_dataset(flags)
  else:
    raise ValueError(f'Unsupported dataset: {flags.dataset}')
