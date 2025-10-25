from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN
import torch
from transformers import Trainer

class ModelWrapper():
  model: FasterRCNN = fasterrcnn_resnet50_fpn_v2()

  def __init__(self, model):
    if model != None:
      self.model = model

  def train(self):
    with torch.no_grad():
      self.model