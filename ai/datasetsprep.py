import os
import cv2
import numpy as np
import json
from utils.preprocessing import preprocess, project_2d, convert_to_yolo, ltp
import random
import joblib
import pickle
import sys

random.seed(42)

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

PATCH_SIZE = (128, 64)
TOTAL_SIZE = 5924
train_limit = int(TRAIN_SPLIT * TOTAL_SIZE)
val_limit = int(VAL_SPLIT * TOTAL_SIZE) + train_limit

new_dataset_path = 'a9_dataset'

class_map = {
	'Truck': 0, 
	'Car': 1, 
	'Van': 2, 
	'Trailer': 3, 
	'Other Vehicles': 4, 
	'Pedestrian': 5, 
}

selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
pca = joblib.load('outputs/hog_pca.joblib')
hog = cv2.HOGDescriptor(PATCH_SIZE, (16, 16), (8, 8), (8, 8), 9)
def extract_features(img):
	# Get bounding boxes using Selective Search
	
	resized_img = cv2.resize(img, PATCH_SIZE)
	
	if len(resized_img.shape) == 2:
			# It is already 2D (Height, Width) -> No conversion needed
			gray_image = resized_img
	elif resized_img.shape[2] == 1:
			# It is 3D but has 1 channel (Height, Width, 1) -> Squeeze it
			gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # Sometimes safe, but better to just squeeze
			gray_image = resized_img.squeeze()
	else:
			# It is BGR (Height, Width, 3) -> Convert
			gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

	feat = []

	# HOG
	hog_feat = hog.compute(gray_image).flatten()
	hog_feat = pca.transform(hog_feat.reshape(1, -1)).flatten()
	feat.extend(hog_feat)

	# HSV
	# hsv_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
	# hsv_hist = cv2.calcHist([hsv_image], [0, 1], None, [30, 16], [0, 180, 0, 256]).flatten()
	# feat.extend(hsv_hist)

	# LTP
	ltp_feat = ltp(gray_image, 10)
	feat.extend(ltp_feat.flatten())

	return np.array(feat)

# xywh format
def compute_iou(box1, box2):
	x1a, y1a, x2a, y2a = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
	x1b, y1b, x2b, y2b = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
	
	intersection_width = min(x2a, x2b) - max(x1a, x1b)
	intersection_height = min(y2a, y2b) - max(y1a, y1b)
	
	if intersection_width <= 0 or intersection_height <= 0: return 0.0
	
	# Calculate union area
	intersection_area = intersection_width * intersection_height
	box1_area = box1[2] * box1[3]
	box2_area = box2[2] * box2[3]
	
	union_area = box1_area + box2_area - intersection_area

	# Calculate IoU
	iou = intersection_area / (union_area + 1e-6)
	return iou

def process_split_and_convert_to_svm_format(dataset_path='datasets', sample_path=None, startindex=0, process_count=100):
	os.makedirs(f'{new_dataset_path}/svm_data', exist_ok=True)

	all_samples = []
	svm_features = []
	svm_labels = []
	gt_boxes = []
	negative_limits = 10

	if not os.path.exists('./svm_samples_cache.npy'):
		for subset in os.listdir(dataset_path):
			if not subset.startswith('a9'): continue
			
			subset_path = os.path.join(dataset_path, subset)
			images_path = os.path.join(subset_path, '_images')
			lables_path = os.path.join(subset_path, '_labels')
		
			if not os.path.exists(images_path): continue
			
			cameras = os.listdir(images_path)
			
			for camera in cameras:
				cam_images_path = os.path.join(images_path, camera)
				cam_labels_path = os.path.join(lables_path, camera)
				if not os.path.exists(cam_images_path): continue
					
				image_files = [f for f in os.listdir(cam_images_path) if f.endswith('.png')]
				for img_name in image_files:
					if not img_name.endswith('.png'): continue
			
					full_image_path = os.path.join(cam_images_path, img_name)
					
					json_name = img_name.replace('.png', '.json')
					full_label_path = os.path.join(cam_labels_path, json_name)
					if os.path.exists(full_label_path):
						all_samples.append((full_image_path, full_label_path))
		random.shuffle(all_samples)
		np.save('./svm_samples_cache.npy', all_samples)
	else:
		all_samples = np.load('./svm_samples_cache.npy')
	
	if startindex is not None:
		all_samples = all_samples[startindex:startindex+process_count]
	
	print(f"Processing {len(all_samples)} images for SVM features...")
	processed_count = 0	

	for full_image_path, full_label_path in all_samples:
		img = cv2.imread(full_image_path)
		if img is None: continue
		orig_h, orig_w = img.shape[:2]

		with open(full_label_path, 'r') as f:
			labels = json.load(f)
		
		orig_h, orig_w = img.shape[:2]

		preprocessed_img = preprocess(img)	
	
		gt_boxes = []
		
		for patch in labels['labels']:
			category = patch.get('category')
			if category not in class_map.keys() or 'box3d_projected' not in patch: continue

			x_min, x_max, y_min, y_max = project_2d(orig_w, orig_h, np.array(list(patch['box3d_projected'].values())))
			w, h = x_max - x_min, y_max - y_min
			if w <= 10 or h <= 10: continue
			gt_boxes.append([x_min, y_min, w, h])
			patch = preprocessed_img[y_min:y_max, x_min:x_max]
			
			if patch.size > 0:
				features = extract_features(patch)
				svm_features.append(features)
				svm_labels.append(class_map.get(category))

		resized_img = cv2.resize(img, (640, 640))
		scale_x = orig_w / 640.0
		scale_y = orig_h / 640.0

		selective_search.setBaseImage(resized_img)
		selective_search.switchToSelectiveSearchFast()

		boxes = selective_search.process()
		np.random.shuffle(boxes)

		neg_count = 0
		for (xs, ys, ws, hs) in boxes:
			if neg_count >= negative_limits: break
			
			# FIX: Scale coordinates UP to original image
			x = int(xs * scale_x)
			y = int(ys * scale_y)
			w = int(ws * scale_x)
			h = int(hs * scale_y)

			proposal_box = [x, y, w, h]
			best_iou = 0.0

			for gt_box in gt_boxes:
				iou = compute_iou(proposal_box, gt_box)

				if iou > best_iou:
					best_iou = iou

			if best_iou < 0.3:
				y2, x2 = min(orig_h, y+h), min(orig_w, x+w)
				patch = preprocessed_img[y:y2, x:x2]
				
				if patch.size > 0:
					features = extract_features(patch)
					svm_features.append(features)
					svm_labels.append(6)
					neg_count += 1
	
		processed_count += 1
		if processed_count % 100 == 0:
			print(f"Processed {processed_count} images...")
	
	data = {
		'features': np.array(svm_features),
		'labels': np.array(svm_labels)
	}

	end_index = startindex + len(svm_labels)
	with open(f'{new_dataset_path}/svm_data/{startindex}-{end_index}_svm_features.pkl', 'wb') as f:
		pickle.dump(data, f)
	
	print('Done!')

def merge_svm_data():
	svm_path = 'a9_dataset/svm_data'
	
	full_features = []
	full_labels = []
	
	batch_files = [f for f in os.listdir(svm_path) if f.endswith('.pkl') and 'features' in f]
	print(f"Found {len(batch_files)} batches to merge...")
	
	for filename in os.listdir(svm_path):
		file_path = os.path.join(svm_path, filename)

		with open(file_path, 'rb') as f:
				ds = pickle.load(f)
		
		full_features.append(ds['features'])
		full_labels.append(ds['labels'])
	
	data = {
		'features': np.vstack(full_features),
		'labels': np.concatenate(full_labels),
	}
	
	output_path = os.path.join(svm_path, 'full_features.pkl')
	with open(output_path, 'wb') as f:
		pickle.dump(data, f)

def process_split_and_convert_to_yolo_format(dataset_path='datasets'):
	split = 'train'

	for s in ['train', 'val', 'test']:
		os.makedirs(f'{new_dataset_path}/yolo_data/{s}/images', exist_ok=True)
		os.makedirs(f'{new_dataset_path}/yolo_data/{s}/labels', exist_ok=True)

	all_samples = []

	for subset in os.listdir(dataset_path):
		if not subset.startswith('a9'): continue
		
		subset_path = os.path.join(dataset_path, subset)
		images_path = os.path.join(subset_path, '_images')
		lables_path = os.path.join(subset_path, '_labels')
	
		if not os.path.exists(images_path): continue
		
		cameras = os.listdir(images_path)
		
		for camera in cameras:
			cam_images_path = os.path.join(images_path, camera)
			cam_labels_path = os.path.join(lables_path, camera)
				
			image_files = [f for f in os.listdir(cam_images_path) if f.endswith('.png')]
			for img_name in image_files:
				if not img_name.endswith('.png'): continue
		
				full_image_path = os.path.join(cam_images_path, img_name)
				
				json_name = img_name.replace('.png', '.json')
				full_label_path = os.path.join(cam_labels_path, json_name)
				if os.path.exists(full_label_path):
					all_samples.append((full_image_path, full_label_path))
		
	random.shuffle(all_samples)
	processed_count = 0	

	for i, (full_image_path, full_label_path) in enumerate(all_samples):
		if i < train_limit:
			split = 'train'
		elif i < val_limit:
			split = 'val'
		else:
			split = 'test'

		img = cv2.imread(full_image_path)
		if img is None: continue
		orig_h, orig_w = img.shape[:2]

		with open(full_label_path, 'r') as f:
			label = json.load(f)
		
		base_name = os.path.splitext(os.path.basename(full_image_path))[0]
		unique_name = f"{i}_{base_name}"
		
		yolo_entries = []
		for patch in label['labels']:
			category = patch.get('category')
			if category not in class_map.keys() or 'box3d_projected' not in patch: continue

			x_min, x_max, y_min, y_max = project_2d(orig_w, orig_h, np.array(list(patch['box3d_projected'].values())))
			nx, ny, nw, nh = convert_to_yolo(x_min, y_min, x_max, y_max, orig_w, orig_h)
			
			yolo_entries.append(f"{class_map.get(category)} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

		img = cv2.resize(img, (640, 640))
		
		orig_img_save_path = f'{new_dataset_path}/yolo_data/{split}/images/{unique_name}.jpg'
		cv2.imwrite(orig_img_save_path, img)

		label_save_path = f'{new_dataset_path}/yolo_data/{split}/labels/{unique_name}.txt'
		if len(yolo_entries) > 0:
			with open(label_save_path, 'w') as f:
				f.write('\n'.join(yolo_entries))
		
		processed_count += 1
		if processed_count % 100 == 0:
			print(f"Processed {processed_count} images...")
		
	print('Done!')

# process_split_and_convert_to_svm_format(sample_path='./svm-samples-temp', startindex=int(sys.argv[1]), process_count=int(sys.argv[2]))
merge_svm_data()