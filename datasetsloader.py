import tensorflow as tf
import os
import cv2
from preprocessing import preprocess_data
import json
from torch.utils.data import Dataset, DataLoader

class TUMDataset(Dataset):
  def __init__(
    self,
    annot_path: str,
    img_dir: str,
    categories_path: str,
    transform=None
  ):
    super().__init__()
    
    with open(annot_path, 'r') as f:
      self.annot = json.load(f)
      
    with open(categories_path, 'r') as f:
      self.categories = json.load(f)
    
    self.img_dir = img_dir
    self.transform = transform
    
    self.rows = []
    for img in self.data["images"]:
      img_id = img["id"]
      annotations = [a for a in self.data["annotations"] if a["image_id"] == img_id]
      self.rows.append({
        "img_path": img["path"],
        "annotations": annotations,
      })
    
  def __len__(self):
    return len(self.rows)

  def __getitem__(self, index):
    rows = self.samples[index]
    img = cv2.imread(rows["img_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if self.transform:
      img = self.transform(img)
    
    return img, rows["annotations"]
  
  def get_categories(self):
    return self.categories