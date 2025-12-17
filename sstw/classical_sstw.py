import numpy as np
import torch
from torch import nn
import cv2
import joblib
from preprocessing import ltp, resize_image_and_bboxes, get_bbox_points
import copy


class FeatureExtractor():
	def __init__(self, pca_path='outputs/hog_pca.joblib', feat_scaler_path='outputs/feat_scaler.joblib'):

		self.region_proposal_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
		strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
		strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
		strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
		
		strategy_combined = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple()
		strategy_combined.addStrategy(strategy_color, 0.5)
		strategy_combined.addStrategy(strategy_texture, 0.5)
		strategy_combined.addStrategy(strategy_size, 0.5)

		self.region_proposal_search.addStrategy(strategy_combined)
	
		self.hog_win_size = (128, 64)
		self.hog_descriptor = cv2.HOGDescriptor(self.hog_win_size, (16, 16), (8, 8), (8, 8), 9)
	
		self.hog_pca = joblib.load(pca_path)
		self.feat_scaler = joblib.load(feat_scaler_path)
	
	def extract_features(self, X):
		X_resized, _ = resize_image_and_bboxes(X, {'target_bboxes': [[0, 0, 0, 0]]})
	
		self.region_proposal_search.setBaseImage(X_resized)
		self.region_proposal_search.switchToSelectiveSearchFast()
		self.region_proposal_search.switchToSingleStrategy(k=100, sigma=0.8)
		boxes = self.region_proposal_search.process()
		
		features_list = []

		for box in boxes:
			feat = []

			x, y, x2, y2 = get_bbox_points(box)
			cropped_roi = X_resized[y:y2, x:x2]
			resized_roi = cv2.resize(cropped_roi, self.hog_win_size)
			gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_RGB2GRAY)

			# HOG
			hog_feat = self.hog_descriptor.compute(gray_roi).flatten()
			hog_feat = self.hog_pca.transform(hog_feat.reshape(1, -1)).flatten()
			feat.extend(hog_feat)
			
			# HSV
			hsv_image = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2HSV)
			hsv_hist = cv2.calcHist([hsv_image], [0, 1], None, [30, 16], [0, 180, 0, 256])
			feat.extend(hsv_hist.flatten())

			# LTP
			ltp_feat = ltp(gray_roi, 10)
			feat.extend(ltp_feat.flatten())

			features_list.append(feat)
	
		boxes_np = np.array(boxes)
		features_np = np.array(features_list)

		if len(features_np) > 0:
			features_np = self.feat_scaler.transform(features_np)

		return boxes_np, features_np

class DetectionHead(nn.Module):
	def __init__(
		self,
		label_num_classes: int=7,
		feature_dim: int=1437,
		**kwargs
	):
		super().__init__(**kwargs)
		self.label_classifier = nn.Sequential(
			nn.Linear(feature_dim, 1024),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.2),

			nn.Linear(512, 128),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.1),

			nn.Linear(128, 64),
			nn.LeakyReLU(0.1),

			nn.Linear(64, label_num_classes + 1),
		)
	
		self.bbox_regressor = nn.Sequential(
			nn.Linear(feature_dim + 4, 1024),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.2),

			nn.Linear(512, 256),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.1),

			nn.Linear(256, 4),
		)

	def forward(self, bbox, complete_features):
		label_logits = self.label_classifier(complete_features)
		bbox_deltas = self.bbox_regressor(torch.cat((complete_features, bbox), dim=-1))

		return label_logits, bbox_deltas

class ClassicalSSTWModel(nn.Module):
	def __init__(
		self,
		num_classes: int=7,
		feature_dim: int=1437,
		**kwargs
	):
		super().__init__(**kwargs)

		self.feature_extractor = FeatureExtractor()
		self.detection_head = DetectionHead(num_classes, feature_dim)
	
	def forward(self, X, gt_boxes=None, training=True):
		
		if training == False:
			bboxes, features = self.feature_extractor.extract_features(X)
		else:
			bboxes, features = gt_boxes["proposed_bboxes"], gt_boxes["proposed_features"]
	
		if len(bboxes) == 0:
			return None, None
		
		device = next(self.detection_head.parameters()).device

		if isinstance(bboxes, torch.Tensor):
			bboxes_tensor = bboxes.float().to(device)
		else:
			bboxes_tensor = torch.from_numpy(bboxes).float().to(device)
				
		if isinstance(features, torch.Tensor):
			features_tensor = features.float().to(device)
		else:
			features_tensor = torch.from_numpy(features).float().to(device)
	
		label_logits, bbox_deltas = self.detection_head(bboxes_tensor, features_tensor)

		if training:
			return label_logits, bbox_deltas
		else:
			probs = torch.softmax(label_logits, dim=1)
			
			max_scores, predicted_classes = torch.max(probs, dim=1)
			
			background_class_id = self.detection_head.label_classifier[-1].out_features - 1
			
			keep_indices = (predicted_classes != background_class_id) & (max_scores > 0.5)
			
			final_boxes = bboxes_tensor[keep_indices]
			final_deltas = bbox_deltas[keep_indices]
			final_labels = predicted_classes[keep_indices]
			final_scores = max_scores[keep_indices]
			
			return final_boxes, final_labels, final_scores, final_deltas

def train(
	model: ClassicalSSTWModel,
	train_loader: torch.utils.data.DataLoader,
	val_loader: torch.utils.data.DataLoader,
	epochs: int,
	device: torch.device,
	save_path: str,
	save_interval: int,
	patience: int,
	early_stopping: bool,
	delta: float,
) -> tuple[dict[str, any], list[float], list[float], list[float], list[float], list[float], list[float]]:
	best_model_params: dict[str, any] = {}
	best_loss = float("inf")
	non_improving = 0
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
	label_loss_fn = torch.nn.CrossEntropyLoss()
	bbox_deltas_loss_fn = torch.nn.SmoothL1Loss()
	
	cls_loss_history = []
	reg_loss_history = []
	loss_history = []
	
	val_cls_loss_history = []
	val_reg_loss_history = []
	val_loss_history = []
	
	for epoch in range(epochs):
		model.train()
		running_train_cls_loss = 0
		running_train_reg_loss = 0
		running_train_loss = 0
		
		for Xs, Ys in train_loader:
			optimizer.zero_grad()
			
			proposed_features = Ys['proposed_features'].to(device)
			proposed_bboxes = Ys['proposed_bboxes'].to(device)
			target_labels = Ys['labels'].to(device)
			target_bbox_deltas = Ys['bbox_deltas'].to(device)
			
			cls_logits, bbox_deltas = model(None, {
				'proposed_bboxes': proposed_bboxes,
				'proposed_features': proposed_features,
			})
			
			cls_loss = label_loss_fn(cls_logits.permute(0, 2, 1), target_labels)
			reg_loss = bbox_deltas_loss_fn(bbox_deltas, target_bbox_deltas)
			
			loss = cls_loss + (5. * reg_loss)
			
			running_train_cls_loss += cls_loss.item()
			running_train_reg_loss += reg_loss.item()
			running_train_loss += loss.item()
			
			loss.backward()
			optimizer.step()
		
		train_count = len(train_loader)
		
		avg_train_loss = running_train_loss / train_count
		
		cls_loss_history.append(running_train_cls_loss / train_count)
		reg_loss_history.append(running_train_reg_loss / train_count)
		loss_history.append(avg_train_loss)
		
		model.eval()
		running_val_cls_loss = 0
		running_val_reg_loss = 0
		running_val_loss = 0
		
		for Xs, Ys in val_loader:
			with torch.no_grad():
				proposed_features = Ys['proposed_features'].to(device)
				proposed_bboxes = Ys['proposed_bboxes'].to(device)
				target_labels = Ys['labels'].to(device)
				target_bbox_deltas = Ys['bbox_deltas'].to(device)
				
				cls_logits, bbox_deltas = model(None, {
					'proposed_bboxes': proposed_bboxes,
					'proposed_features': proposed_features,
				})
				
				cls_loss = label_loss_fn(cls_logits.permute(0, 2, 1), target_labels)
				reg_loss = bbox_deltas_loss_fn(bbox_deltas, target_bbox_deltas)
				
				loss = cls_loss + (5. * reg_loss)
				
				running_val_cls_loss += cls_loss.item()
				running_val_reg_loss += reg_loss.item()
				running_val_loss += loss.item()
		
		val_count = len(val_loader)
		
		avg_val_loss = running_val_loss / val_count
		
		val_cls_loss_history.append(running_val_cls_loss / val_count)
		val_reg_loss_history.append(running_val_reg_loss / val_count)
		val_loss_history.append(avg_val_loss)
		
		if epoch % save_interval == 0:
			torch.save(model.state_dict(), save_path)

		if avg_val_loss < best_loss - delta:
			best_loss = avg_val_loss
			best_model_params = copy.deepcopy(model.state_dict())
			
			torch.save(model.state_dict(), save_path.replace(".pth", "_best.pth"))
			
			non_improving = 0
		else:
			non_improving += 1
		
		if non_improving >= patience:
			if early_stopping:
				break
			non_improving = 0
		
		print(f"Epoch [{epoch+1}/{epochs}] - Average Validation Loss: {avg_val_loss:.2f}, Average Training Loss: {avg_train_loss:.2f}")

	return best_model_params, cls_loss_history, reg_loss_history, loss_history, val_cls_loss_history, val_reg_loss_history, val_loss_history

print('Imported!')