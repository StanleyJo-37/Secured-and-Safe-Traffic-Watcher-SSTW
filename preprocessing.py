import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
import PIL.Image as Image

DATASET_PATH = "./TUM"
FRAME_PATH = "./FRAMES"

toTensor = torchvision.transforms.Compose([
  torchvision.transforms.Resize((224, 224)),
  torchvision.transforms.ToTensor(),
])

def to_frames(dataset_path: str=DATASET_PATH, frame_path: str=FRAME_PATH) -> None:
  """
    Convert videos into frames and save them into a directory,
    ready to be processed in later pipeline.
  """

  for name in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, name)
    save_dir = os.path.join(frame_path, os.path.splitext(name)[0])
    os.makedirs(save_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    count, success = 0, True
    while success:
      success, image = video.read()

      if success:
        cv2.imwrite(f"{frame_path}/{name}/frame{count}.jpg", image)
        count += 1
    
    video.release()

# def remove_glare(input_frame, patch_size):
#   gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
#   mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

#   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



#   return image

def normalize(input_frame):
  input_frame = cv2.GaussianBlur((3, 3), 0)
  input_frame = cv2.convertScaleAbs(input_frame, alpha=1.2)
  input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
  # input_frame = remove_glare(input_frame)
  return input_frame

def to_tensors(frames_path: str) -> List[torch.Tensor]:
  """
    Convert frames from a directory into a list of tensors,
    comprising of one whole video.
  """

  tensors: List[torch.Tensor] = [[]]

  for frame_name in sorted(os.listdir(frames_path)):
    full_path = os.path.join(frames_path, frame_name)
    
    frame = cv2.imread(full_path, flags=cv2.IMREAD_COLOR)
    frame = normalize(frame)
    array = Image.fromarray(frame)
    
    tensor = toTensor(array)
    tensors.append(tensor)

  return tensors


