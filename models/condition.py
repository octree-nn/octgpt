import torch
import torch.nn as nn
from typing import Union
from .networks import resnet_model, clip_model
import timm
import kornia


class ImageEncoder(nn.Module):
  def __init__(self, encoder_type: Union[str, nn.Module]):
    super().__init__()
    self.encoder_type = encoder_type
    if isinstance(encoder_type, str):
      if encoder_type == "resnet":
        self.encoder = resnet_model.resnet18(pretrained=True).float()
      elif encoder_type == "vit":
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
        )
      elif encoder_type == "clip":
        self.encoder = clip_model.CLIPEncoder()
      else:
        raise ValueError(f"Unknown encoder type: {
                         encoder_type}, choose from ['resnet', 'vit', 'clip']")
    elif isinstance(encoder_type, nn.Module):
      self.encoder = encoder_type
    else:
      raise ValueError(f"Unknown encoder type: {encoder_type}")

  def resnet_prep(self, x):
    x = kornia.geometry.center_crop(
        x, (224, 224), mode='bilinear', align_corners=True)
    x = (x + 1) / 2
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    x = kornia.enhance.normalize(x, mean, std)
    return x

  def clip_prep(self, x):
    x = kornia.geometry.center_crop(
        x, (224, 224), mode='bicubic', align_corners=True)
    x = (x + 1) / 2
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
    x = kornia.enhance.normalize(x, mean, std)
    return x

  def forward(self, image=None):
    c_mm = []
    image = image[:, :3]
    if self.encoder_type == 'resnet':
      bs = image.shape[0]
      image = self.resnet_prep(image)
      c_image = self.encoder(image)
      p_image = torch.rand(bs, device=image.device) > 0.5
      c_image = c_image * p_image[:, None, None]
      c_mm.append(c_image)
    elif self.encoder_type == 'clip':
      image = self.clip_prep(image)
      c_image = self.encoder(image, None)
      c_mm += c_image
    elif self.encoder_type == 'vit':
      c_image = self.clip_prep(image)
      c_image = self.encoder.forward_features(c_image)
      c_image = c_image[:, :-1]
      c_mm.append(c_image)
    return c_mm
