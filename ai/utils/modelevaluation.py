import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch import nn
from model import Model
from torch.utils.data import DataLoader

def get_eval(
  model: Model,
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
  model: Model,
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
  model: Model,
  path: str
):
  torch.save(model.state_dict(), path)

def load(
  model: Model,
  path: str
):
  model.load_state_dict(torch.load(path), strict=False)
