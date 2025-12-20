import os
import cv2
import numpy as np
import json
from utils.preprocessing import preprocess, project_2d, convert_to_yolo

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

TOTAL_SIZE = 5924
train_limit = int(TRAIN_SPLIT * TOTAL_SIZE)
val_limit = int(VAL_SPLIT * TOTAL_SIZE) + train_limit

processed_count = 0

new_dataset_path = 'a9_dataset'

labels = {
	'Truck': 0, 
	'Car': 1, 
	'Van': 2, 
	'Trailer': 3, 
	'Other Vehicles': 4, 
	'Pedestrian': 5, 
}

# def store_svm_patches(img, labels):
#   pass

def process_split_and_convert_to_yolo_format(dataset_path='datasets', include_svm=False):
	global processed_count
	split = 'train'
	
	for subset in os.listdir(dataset_path):
		if not subset.startswith('a9'):
			continue
		
		subset_path = os.path.join(dataset_path, subset)
		images_path = os.path.join(subset_path, '_images')
		lables_path = os.path.join(subset_path, '_labels')
  
		if not os.path.exists(images_path): continue
		
		cameras = os.listdir(images_path)
		
		for camera in cameras:
			cam_images_path = os.path.join(images_path, camera)
			cam_labels_path = os.path.join(lables_path, camera)

			for s in ['train', 'val', 'test']:
				os.makedirs(f'{new_dataset_path}/yolo_data/{s}/images', exist_ok=True)
				os.makedirs(f'{new_dataset_path}/yolo_data/{s}/labels', exist_ok=True)
				# if include_svm:
				# 		os.makedirs(f'{new_dataset_path}/{s}/svm_data', exist_ok=True)
				
			image_files = [f for f in os.listdir(cam_images_path) if f.endswith('.png')]
			for img_name in image_files:
				
				if processed_count < train_limit: split = 'train'
				elif processed_count < val_limit: split = 'val'
				else: split = 'test'
    
				full_image_path = os.path.join(cam_images_path, img_name)
				
				json_name = img_name.replace('.png', '.json')
				full_label_path = os.path.join(cam_labels_path, json_name)
    
				if not os.path.exists(full_label_path): continue

				img = cv2.imread(full_image_path)
				if img is None: continue

				orig_h, orig_w = img.shape[:2]
    
				label = json.load(open(full_label_path, 'r'))
				
				base_name = os.path.splitext(img_name)[0]
				
				yolo_lines = []
				
				for patch in label['labels']:
					category = patch.get('category')
					if category not in labels.keys():
						continue

					x_min, x_max, y_min, y_max = project_2d(orig_w, orig_h, np.array(list(patch['box3d_projected'].values())))
					nx, ny, nw, nh = convert_to_yolo(x_min, y_min, x_max, y_max, orig_w, orig_h)
					
					category_id = labels.get(category)
					line = f"{category_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
					yolo_lines.append(line)

				img = cv2.resize(img, (640, 640))

				# if include_svm:
				# 	store_svm_patches(img, label['labels'])
				
				orig_img_save_path = f'{new_dataset_path}/yolo_data/{split}/images/{base_name}.jpg'
				cv2.imwrite(orig_img_save_path, img)
    
				label_save_path = f'{new_dataset_path}/yolo_data/{split}/labels/{base_name}.txt'
				if len(yolo_lines) > 0:
					with open(label_save_path, 'w') as f:
						f.write('\n'.join(yolo_lines))
				
				processed_count += 1
				if processed_count % 50 == 0:
					print(f"Processed {processed_count} images...")
		
	print('Done!')
	
process_split_and_convert_to_yolo_format()