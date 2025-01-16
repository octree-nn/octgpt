import torch
import torch.nn as nn
from typing import Union
import transformers
class ImageEncoder(nn.Module):
  def __init__(self, encoder_type: str):
    super().__init__()
    self.encoder_type = encoder_type
    if encoder_type == "vit":
      VIT_MODEL_NAME = "google/vit-large-patch16-224-in21k"
      self.processor  = transformers.ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
      self.encoder = transformers.ViTModel.from_pretrained(VIT_MODEL_NAME)
    elif encoder_type == "dinov2":
      DINO_MODEL_NAME = "facebook/dinov2-large"
      self.processor  = transformers.AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
      self.encoder = transformers.AutoModel.from_pretrained(DINO_MODEL_NAME)
    elif encoder_type == "clip":
      CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
      self.processor  = transformers.CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
      self.encoder = transformers.CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
    else:
      raise ValueError(f"Unknown encoder type: {encoder_type}")

  def forward(self, images, device):
    if self.encoder_type == 'vit':
      images = self.processor(images, return_tensors="pt").to(device)
      outputs = self.encoder(**images)
      features = outputs.last_hidden_state # (B, 197, 1024)
    elif self.encoder_type == "dinov2":
      images = self.processor(images, return_tensors="pt").to(device)
      outputs = self.encoder(**images)
      features = outputs.last_hidden_state # (B, 257, 1024)
    elif self.encoder_type == "clip":
      images = self.processor(images, return_tensors="pt").to(device)
      outputs = self.encoder(**images)
      features = outputs.last_hidden_state # (B, 257, 1024)
    return features

class TextEncoder(nn.Module):
  def __init__(self, encoder_type: str):
    super().__init__()
    self.encoder_type = encoder_type
    if encoder_type == "clip":
      CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
      self.processor  = transformers.CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
      self.encoder = transformers.CLIPTextModel.from_pretrained(CLIP_MODEL_NAME)
    else:
      raise ValueError(f"Unknown encoder type: {encoder_type}")

  def forward(self, text, device):
    if self.encoder_type == "clip":
      text = self.processor(text=text, return_tensors="pt", max_length=77).to(device)
      outputs = self.encoder(**text)
      features = outputs.last_hidden_state # (B, 77, 768)
    return features
