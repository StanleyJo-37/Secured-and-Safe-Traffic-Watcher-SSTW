import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch import nn
from torch.utils.data import DataLoader

# xywh format
def compute_iou(box1, box2):
  x1a, y1a, x2a, y2a = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
  x1b, y1b, x2b, y2b = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
  
  intersection_width = min(x2a, x2b) - max(x1a, x1b)
  intersection_height = min(y2a, y2b) - max(y1a, y1b)
  
  if intersection_width <= 0 or intersection_height <= 0:
    return 0
  
  intersection_area = intersection_width * intersection_height

  # Calculate union area
  box1_area = box1[2] * box1[3]
  box2_area = box2[2] * box2[3]
  
  union_area = box1_area + box2_area - intersection_area

  # Calculate IoU
  iou = intersection_area / union_area
  return iou

def get_eval(
  model: nn.Module,
  test_dataset: DataLoader,
  iou_threshold=0.8
):
  model.eval()

  mAP = MeanAveragePrecision()
  true_positive, false_positve, false_negative = 0, 0, len(outputs)
  total_loss, total_samples = 0, 0
  
  test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

  with torch.no_grad():

    for images, targets in test_loader:
      outputs = model(images)
      mAP.update(outputs, targets)

      for ann in targets:
        matched = False
        for out in outputs:
          iou = compute_iou(out[:4], ann)

          if iou > iou_threshold:
            matched = True
            break
        
        if matched:
          true_positive += 1
        else:
          false_positve += 1
        
        false_negative -= 1

    mAP_eval = mAP.compute()

  accuracy  = true_positive + (false_positve + false_negative)
  precision = true_positive + (true_positive + false_positve)
  recall    = true_positive + (true_positive + false_negative)
  f1_score  = (2 * precision * recall) / (precision + recall)

  return mAP_eval, accuracy, precision, recall, f1_score

def print_eval(
  model: nn.Module,
  test_dataset: torch.utils.data.DataLoader,
  iou_threshold=0.8
):
  mAP_eval, accuracy, precision, recall, f1_score = get_eval(
    model,
    test_dataset,
    iou_threshold
  )

  print(f"Mean Average Precision: {mAP_eval}")
  print(f"Accuracy              : {accuracy}")
  print(f"Precision             : {precision}")
  print(f"Recall                : {recall}")
  print(f"F1 Score              : {f1_score}")

def save(
  model: nn.Module,
  path: str
):
  torch.save(model.state_dict(), path)

def load(
  model: nn.Module,
  path: str
):
  model.load_state_dict(torch.load(path), strict=False)
