import cv2
import numpy as np
import joblib
from dcp.haze_remover import HazeRemover

def get_bbox_points(bbox):
	x, y, w, h = bbox
	x2 = x + w
	y2 = y + h

	return max(0, int(x)), max(0, int(y)), x2, y2

def resize_image_and_bboxes(image, labels, scale=0.25):
	if scale == 1.0:
		return image, labels

	resized_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA)

	new_bboxes = []

	for bbox in labels['target_bboxes']:
		new_bboxes.append([int(c * scale) for c in bbox])

	return resized_img, { **labels, 'target_bboxes': new_bboxes }

uniform_lookup = np.array([58] * 256)
bin_id = 0

def uniformity(bits):
	transitions = 0
	curr_bit = bits[0]

	for bit in bits[1:]:
		if bit != curr_bit:
			transitions += 1
		curr_bit = bit

	return transitions

for i in range(256):
	bits = np.array([(i >> j) & 1 for j in range(8)])

	if uniformity(bits) <= 2:
		uniform_lookup[i] = bin_id
		bin_id += 1
	else:
		uniform_lookup[i] = 58

def bits_to_integer(bits: list) -> int:
	total = 0

	for i, bit in enumerate(bits):
		total += pow(2, i) * bit

	return total

def ltp(img, k):
	hist_upper = np.zeros(59, int)
	hist_lower = np.zeros(59, int)

	offsets = [
		(-1, -1), (-1, 0), (-1, 1),
		(0, -1),           (0, 1),
		(1, -1), (1, 0), (1, 1),
	]

	for i in range(1, img.shape[0] - 1):
		for j in range(1, img.shape[1] - 1):

			upper_bits, lower_bits = [], []
			center = img[i, j]

			center = int(center)
			k = int(k)

			lower_bound = max(0, center - k)
			upper_bound = min(255, center + k)

			for dy, dx in offsets:
				val = img[i + dy, j + dx]

				upper_bits.append(val > upper_bound)
				lower_bits.append(val < lower_bound)

			hist_upper[uniform_lookup[bits_to_integer(upper_bits)]] += 1
			hist_lower[uniform_lookup[bits_to_integer(lower_bits)]] += 1

	return np.concatenate([hist_upper, hist_lower])

def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def is_foggy(image):
  var_l = variance_of_laplacian(image)
  return var_l < 50

haze_remover = HazeRemover()
clahe = cv2.createCLAHE(2.0, (8, 8))

def preprocess(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gray_final = cv2.medianBlur(img_gray, 3)
  
  if is_foggy(img_gray_final):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_dehazed = haze_remover.remove_haze(img_rgb)
    img_dehazed = np.clip(img_dehazed, 0, 255).astype(np.uint8)
    
    img_gray_final = cv2.cvtColor(img_dehazed, cv2.COLOR_RGB2GRAY)
  
  img_clahe = clahe.apply(img_gray_final)
  
  return img_clahe.astype(np.float32)

hog = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9)
def extract_features(X, pca_path='outputs/hog_pca.joblib'):
	# Get bounding boxes using Selective Search
	selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	selective_search.setBaseImage(X)
	selective_search.switchToSelectiveSearchFast()

	strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
	strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
	strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()

	strategy_combined = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple()
	strategy_combined.addStrategy(strategy_color, 0.5)
	strategy_combined.addStrategy(strategy_texture, 0.5)
	strategy_combined.addStrategy(strategy_size, 0.5)

	selective_search.addStrategy(strategy_combined)
	selective_search.switchToSingleStrategy(k=100, sigma=0.8)

	pca = joblib.load(pca_path)

	boxes = selective_search.process()

	# Feature extraction
	boxes_list = []
	features_list = []
	for box in boxes:
		feat = []

		box = get_bbox_points(box)
		x, y, x2, y2 = box
		cropped_img = X[y:y2, x:x2]
		resized_img = cv2.resize(cropped_img, (128, 64))
		gray_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

		# HOG
		hog_feat = hog.compute(gray_image).flatten()
		hog_feat = pca.transform(hog_feat.reshape(1, -1)).flatten()
		feat.extend(hog_feat)

		# HSV
		hsv_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2HSV)
		hsv_hist = cv2.calcHist([hsv_image], [0, 1], None, [30, 16], [0, 180, 0, 256]).flatten()
		feat.extend(hsv_hist)

		# LTP
		ltp_feat = ltp(gray_image, 10)
		feat.extend(ltp_feat.flatten())

		boxes_list.append(box)
		features_list.append(feat)

	return np.array(boxes_list), np.array(features_list)
