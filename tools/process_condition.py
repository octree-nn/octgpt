import os
import torch
import ocnn
import time
from thsolver import Solver
from thsolver.tracker import AverageTracker
from ognn.octreed import OctreeD
from utils import utils, builder
from utils.expand_ckpt import expand_checkpoint
from utils.distributed import get_rank
from models.mar import MAR, MAREncoderDecoder
from models.condition import ImageEncoder
from datasets import get_shapenet_dataset
from datasets.shapenet_utils import snc_synth_id_to_label_5, category_5_to_num
from tqdm import tqdm
import copy
import cv2


class CondSolver(Solver):
  def __init__(self, FLAGS, is_master=True):
    super().__init__(FLAGS, is_master)
    self.condition_type = FLAGS.MODEL.GPT.condition_type

  def get_model(self, flags):
    if self.condition_type == "image":
      model = ImageEncoder(flags.GPT.condition_encoder)
      model.cuda(device=self.device)
      model.eval()
      utils.set_requires_grad(self.cond_enc, False)
    return model

  def get_dataset(self, flags):
    return get_shapenet_dataset(flags)

  def batch_to_cuda(self, batch):
    if self.condition_type == "image":
      images = batch['image']
      cond = self.model(images, device=self.device)
      batch['condition'] = cond
    else:
      raise NotImplementedError("Condition type not implemented")

  def model_forward(self, batch):
    if self.condition_type == "image":
      images = batch['image']
      cond = self.model(images, device=self.device)
      batch['condition'] = cond
    else:
      raise NotImplementedError("Condition type not implemented")
    output = {'loss': torch.zeros(1).to(self.device)}
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    with torch.no_grad():
      output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output


if __name__ == '__main__':
  CondSolver.main()
