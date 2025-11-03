import cv2
import numpy as np
import matplotlib.pyplot as plt

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
max_V = high_V - 25

def normalize(input_frame):
  input_frame = cv2.convertScaleAbs(input_frame, alpha=1.2).astype(np.float32) / 255.0
  return input_frame

def unsharp_mask(
  image,
  sigma: int = 5,
  strength: float = 0.7
):
  image = image.astype(np.float32)
  
  medianBlurred = cv2.medianBlur(image.astype(np.uint8), sigma)
  
  lap = cv2.Laplacian(medianBlurred, cv2.CV_32F)
  
  sharp = image - strength * lap
  sharp = np.clip(sharp, 0, 255).astype(np.uint8)
  
  return sharp

def bright_pixel_to_dark_pixel_ratio(frame):
  grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  total_pixel_count = frame.shape[0] * frame.shape[1]
  
  dark_pixels = cv2.adaptiveThreshold(
    grayscale_frame,
    100,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
  )
  dark_pixel_count = cv2.countNonZero(dark_pixels)
  bright_pixel_count = total_pixel_count - dark_pixel_count
  
  return bright_pixel_count / dark_pixel_count

def dim_night_light(frame):
  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
  dimmed_frame = cv2.inRange(hsv_frame, (low_H, low_S, low_V), (high_H, high_S, max_V))
  dimmed_frame = cv2.cvtColor(dimmed_frame, cv2.COLOR_HSV2RGB)
  return dimmed_frame

def preprocess_data(frame, label):
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  if bright_pixel_to_dark_pixel_ratio < 0.3:
    frame = dim_night_light(frame)
  
  normalized_frame = normalize(frame)
  sharpened_frame = unsharp_mask(normalized_frame)
  
  return sharpened_frame, label

def test_preprocess():
  orig_img = cv2.imread("./datasets/a9_dataset_r01_s01/_images/s040_camera_basler_north_16mm/1607511137_552725296_s040_camera_basler_north_16mm.png")
  sharpened = unsharp_mask(orig_img)

  plt.figure(figsize=(20, 8))
  plt.axis('off')

  plt.subplot(1, 2, 1)
  plt.imshow(orig_img)
  plt.title("Original Image")

  plt.subplot(1, 2, 2)
  plt.imshow(sharpened)
  plt.title("Sharpened Image")

  plt.tight_layout()
  plt.show()
