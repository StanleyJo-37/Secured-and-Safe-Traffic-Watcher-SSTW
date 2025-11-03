import os
import cv2
import json
from tqdm import tqdm
import numpy as np

DATASET_PATH = "./datasets"

def project_2d(
  bbox_3d: np.array,
):

  xmin = np.min(bbox_3d[:, 0])
  xmax = np.max(bbox_3d[:, 0])
  ymin = np.min(bbox_3d[:, 1])
  ymax = np.max(bbox_3d[:, 1])

  return xmin, xmax, ymin, ymax

image_id = 1
label_id = 1
category_id = 1

category_path = f"{DATASET_PATH}/categories/category.json"
categories: list[dict[str, any]] = []

for path in tqdm(os.listdir(DATASET_PATH)):
  tqdm.write(f"Processing dataset: {path}")
  label_path = f"{DATASET_PATH}/{path}/_labels"
  
  for dir_idx, dir_name in enumerate(os.listdir(label_path)):
    new_label_obj = {
      "images": [],
      "annotations": [],
    }
    
    new_label_dir = f"{DATASET_PATH}/{path}/_new_labels/{dir_name}"
    os.makedirs(new_label_dir, exist_ok=True)
    
    calibration_path = f"{DATASET_PATH}/{path}/_calibration/{dir_name}.json"
    calibration_obj = json.load(open(calibration_path, "r"))
    
    for label_name in os.listdir(f"{label_path}/{dir_name}"):
      full_label_path = f"{label_path}/{dir_name}/{label_name}"
      new_label_path = f"{DATASET_PATH}/{path}/_new_labels/{dir_name}/{label_name}"
      img_path = f"{DATASET_PATH}/{path}/_images/{dir_name}/{label_name.split('.')[0]}.png"
      
      with open(full_label_path, "r") as f:
        label_obj: dict[str, any] = json.load(f)
        
        img = cv2.imread(img_path)
        if img is None:
          print(f"Image not found for {img_path}")
          continue
        
        new_label_obj["images"].append({
          "id": image_id,
          "file_name": label_obj["image_file_name"],
          "width": img.shape[1],
          "height": img.shape[0],
          "path": img_path,
        })
        
        for label in label_obj["labels"]:
          cat_id = next(
            (category["id"] for category in categories
              if category["name"] == label["category"]),
            None
          ) if len(categories) > 0 else None
          
          if cat_id == None:
            categories.append({
              "id": category_id,
              "name": label["category"],
            })
            
            cat_id = category_id
            category_id += 1
            
          xmin, xmax, ymin, ymax = project_2d(
            np.array(list(label["box3d_projected"].values())),
          )
          bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
          
          new_label_obj["annotations"].append({
            "id": label_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "attributes": label.get("attributes", {}),
          })
          
          label_id += 1
        
        image_id += 1
            
    new_label_path = os.path.join(new_label_dir, f"{dir_name}.json")
    with open(new_label_path, "w") as f:
      json.dump(new_label_obj, f, indent=4)

os.makedirs(os.path.dirname(category_path), exist_ok=True)
with open(category_path, "w") as f:
  json.dump(categories, f, indent=4)