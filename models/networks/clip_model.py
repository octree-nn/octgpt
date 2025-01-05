""" 
    Reference:
        - https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py
        - https://github.com/openai/CLIP
"""

import kornia
from einops import rearrange, repeat
from torchvision.transforms import Resize, Normalize

import torch
import torch.nn as nn

import clip


class CLIPEncoder(nn.Module):
  def __init__(
      self,
      model="ViT-B/32",
      jit=False,
      device='cuda' if torch.cuda.is_available() else 'cpu',
      antialias=False,
  ):
    super().__init__()
    self.model, _ = clip.load(name=model, device=device,
                              jit=jit, download_root="./saved_ckpt")
    self.device = device

    # self.model, self.preprocess = clip.load(name=model, device=device, jit=jit)
    self.model = self.model.float()  # turns out this is important...

    self.antialias = antialias

  def forward(self, image=None, text=None):
    # x is assumed to be in range [-1,1]
    out = []
    if image is not None:
      out.append(self.model.encode_image(image).unsqueeze(1))
    if text is not None:
      text = clip.tokenize(text).to(self.device)
      out.append(self.model.encode_text(text).unsqueeze(1))
    return out
