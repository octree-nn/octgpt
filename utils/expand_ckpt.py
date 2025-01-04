import torch
import torch.nn as nn
import copy
from collections import OrderedDict


def expand_checkpoint(model, checkpoint):
  """
  Expand the checkpoint parameters based on the model's dimensions and rules from the table.

  Args:
      model (nn.Module): The PyTorch model to align dimensions with.
      checkpoint (dict): A dictionary containing checkpoint parameters.

  Returns:
      dict: A new checkpoint dictionary with expanded parameters.
  """
  expanded_checkpoint = OrderedDict()
  for name, param in model.named_parameters():
    if name in checkpoint:
      ckpt_param = checkpoint[name]
      if "qkv" in name:
        new_param = expand_qkv(param, ckpt_param)
      elif "rope" in name:
        new_param = expand_mlp(param, ckpt_param, zero_init=False)
      elif "norm" in name and "weight" in name:
        new_param = expand_norm_weight(param, ckpt_param)
      else:
        new_param = expand_mlp(param, ckpt_param)
      # Ensure the expanded parameter matches the model's parameter shape
      assert new_param.shape == param.shape, f"Shape mismatch after expansion: {
          new_param.shape} vs {param.shape}"
      expanded_checkpoint[name] = new_param
    else:  # If the parameter is not in the checkpoint, then initialize with zeros
      expanded_checkpoint[name] = torch.zeros_like(param)
  return expanded_checkpoint


def expand_qkv(param, ckpt_param):
  model_shape = param.shape
  ckpt_shape = ckpt_param.shape
  new_param = torch.zeros_like(param)
  # Case 1: qkv Weight
  if len(ckpt_shape) == 2:
    num_embed_model = model_shape[1]
    num_embed_ckpt = ckpt_shape[1]
    new_param = new_param.reshape(3, num_embed_model, num_embed_model)
    ckpt_param = ckpt_param.reshape(3, num_embed_ckpt, num_embed_ckpt)
    new_param[:, :num_embed_ckpt, :num_embed_ckpt] = ckpt_param
    new_param = new_param.reshape(model_shape)
  elif len(ckpt_shape) == 1:
    num_embed_model = model_shape[0] // 3
    num_embed_ckpt = ckpt_shape[0] // 3
    new_param = new_param.reshape(3, num_embed_model)
    ckpt_param = ckpt_param.reshape(3, num_embed_ckpt)
    new_param[:, :num_embed_ckpt] = ckpt_param
    new_param = new_param.reshape(model_shape)
  return new_param


def expand_mlp(param, ckpt_param, zero_init=True):
  ckpt_shape = ckpt_param.shape
  if zero_init:
    new_param = torch.zeros_like(param)
  else:
    new_param = param.clone().detach()
  # Case 1: Weight
  if len(ckpt_shape) == 2:
    new_param[:ckpt_shape[0], :ckpt_shape[1]] = ckpt_param
  # Case 2: Bias
  elif len(ckpt_shape) == 1:
    new_param[:ckpt_shape[0]] = ckpt_param
  return new_param

def expand_norm_weight(param, ckpt_param):
  model_shape = param.shape
  ckpt_shape = ckpt_param.shape
  new_param = torch.zeros_like(param)
  new_param[:ckpt_shape[0]] = ckpt_param * torch.sqrt(torch.tensor(ckpt_shape[0] / model_shape[0]))
  return new_param