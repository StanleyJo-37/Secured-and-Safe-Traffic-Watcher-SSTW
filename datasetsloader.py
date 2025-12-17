import json
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
import torch
from preprocessing import resize_image_and_bboxes, get_bbox_points

class TUMDataset(Dataset):
	def __init__(
		self,
		annot_paths: list[str],
		categories_path: str,
		features_root: str="./datasets/_extracted_features",
		skip_no_features=True,
		transform=None,
		*args,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		
		self.features_root = features_root
		
		self.annots = []
		for annot_path in annot_paths:
			if "dataset" in annot_path:
				for file_name in os.listdir(annot_path):
					with open(os.path.join(annot_path, file_name), 'r') as f:
						self.annots.append(json.load(f))
			
		with open(categories_path, 'r') as f:
			self.categories = json.load(f)
		
		self.transform = transform
		
		self.data = []
		for annot in self.annots:
			for img in annot["images"]:
				img_name = img["file_name"]
				feature_path = os.path.join(self.features_root, f"{img_name.split('.')[0]}.npy")
				
				if skip_no_features and not os.path.exists(feature_path):
					continue
				img_id = img["id"]
				annotations = [a for a in annot["annotations"] if a["image_id"] == img_id]
				self.data.append({
					"img_path": img["path"],
					"annotations": annotations,
				})
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data = self.data[index]
		img_path = data["img_path"]
		img_name = img_path.split('/')[-1]
		
		img = torchvision.io.decode_image(img_path)
		
		feature_path = os.path.join(self.features_root, f"{img_name.split('.')[0]}.npy")
		
		if os.path.exists(feature_path):
			raw_data = np.load(feature_path)
			if raw_data.size > 0:
				proposal_boxes = raw_data[:, :4] 
				proposal_features = raw_data[:, 4:]
			else:
				proposal_boxes = np.empty((0, 4))
				proposal_features = np.empty((0, 0))
		else:
			proposal_boxes = np.empty((0, 4))
			proposal_features = np.empty((0, 0))
			
		gt_boxes = np.array([[c * 0.25 for c in a["bbox"]] for a in data["annotations"]])
		gt_labels = np.array([a["category_id"] for a in data["annotations"]])
		gt_xyxy = torch.tensor(gt_boxes).float()
		
		sampled_features = []
		sampled_deltas = []
		sampled_labels = []
		sampled_bboxes = []
	
		TARGET_SIZE = 128  # Fixed Batch Size per Image
		feature_dim = proposal_features.shape[1] if len(proposal_features) > 0 else 1437
		
		if len(proposal_boxes) and len(gt_boxes):
			positive_examples = []
			negative_examples = []
			
			props_xyxy = torch.tensor(np.stack([proposal_boxes[:, 0], proposal_boxes[:, 1], proposal_boxes[:, 2], proposal_boxes[:, 3]], axis=1)).float()
			gt_xyxy = torch.tensor(np.stack([gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]], axis=1)).float()
			
			iou_matrix = torchvision.ops.box_iou(props_xyxy, gt_xyxy)
			
			max_ious, best_gt_indices = torch.max(iou_matrix, dim=1)
			
			max_ious = max_ious.numpy()
			best_gt_indices = best_gt_indices.numpy()
			
			# Positives: IoU >= 0.5
			positive_examples = np.where(max_ious >= 0.5)[0]
			
			# Negatives: IoU < 0.3
			negative_examples = np.where(max_ious < 0.3)[0]
		
			sample_nums = 196
			positive_count = int(sample_nums * 0.6)
			
			selections = []
			
			if len(positive_examples) > 0:
				selections.append(np.random.choice(positive_examples, size=min(len(positive_examples), positive_count), replace=False))

			negative_count = sample_nums - positive_count
			if len(negative_examples) > 0:
				selections.append(np.random.choice(negative_examples, size=min(len(negative_examples), negative_count), replace=sample_nums < negative_count))

			if len(selections) > 0:
				final_indices = np.concatenate(selections)

				if len(final_indices) > TARGET_SIZE:
					final_indices = final_indices[:TARGET_SIZE]
    
				np.random.shuffle(final_indices)
				
				for idx in final_indices:
					sampled_features.append(proposal_features[idx])
					
					if idx in positive_examples:
						
						gt_idx = best_gt_indices[idx]
						label = gt_labels[gt_idx]
						
						px, py, px2, py2 = proposal_boxes[idx]
						pw = px2 - px
						ph = py2 - py

						gx, gy, gx2, gy2 = gt_boxes[gt_idx]
						gw = gx2 - gx
						gh = gy2 - gy

						tx = (gx - px) / pw
						ty = (gy - py) / ph
						tw = np.log(gw / pw)
						th = np.log(gh / ph)
						
						sampled_labels.append(label)
						sampled_deltas.append([tx, ty, tw, th])
						sampled_bboxes.append(proposal_boxes[idx])
					else:
						sampled_labels.append(7)
						sampled_deltas.append([0, 0, 0, 0])
						sampled_bboxes.append([0, 0, 0, 0])
			
			np.random.shuffle(selections)

		current_count = len(sampled_labels)
		pad_count = TARGET_SIZE - current_count
  
		if pad_count > 0:
			sampled_features.extend(torch.zeros((pad_count, feature_dim)))
			sampled_deltas.extend(torch.zeros((pad_count, 4)))
			sampled_labels.extend([7] * pad_count)
			sampled_bboxes.extend(torch.zeros((pad_count, 4)))
		
		if len(sampled_features) == 0:
			feature_dim = proposal_features.shape[1] if len(proposal_features) > 0 else 1437
			sampled_features = np.zeros((1, feature_dim))
			sampled_labels = [7]
			sampled_deltas = [[0,0,0,0]]
		
		target = {
			"proposed_features": torch.tensor(np.array(sampled_features), dtype=torch.float32),
			"proposed_bboxes": torch.tensor(np.array(sampled_bboxes), dtype=torch.long),
			"labels": torch.tensor(np.array(sampled_labels), dtype=torch.long),
			"bbox_deltas": torch.tensor(np.array(sampled_deltas), dtype=torch.float32)
		}
		
		if self.transform:
			img = self.transform(img)
		
		return img, target
	
	def get_categories(self) -> list[dict[str, str]]:
		return self.categories
