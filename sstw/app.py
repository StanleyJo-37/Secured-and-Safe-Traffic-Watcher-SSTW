from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision
import os
import cv2
import numpy as np
from classical_sstw import ClassicalSSTWModel
# Import the Tracker wrapper we created previously
# If you haven't saved it to a file, you can paste the class definition here.
from tracker import TrafficTracker 

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. SETUP MODEL & TRACKER ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Model
print("Loading model...")
model = ClassicalSSTWModel(num_classes=7) # Ensure this matches training
try:
    # Load weights
    model.load_state_dict(torch.load('outputs/best_model_new.pth', map_location=device))
    model.to(device)
    model.eval()
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model_loaded = False
    model = None

# Initialize Tracker (Global instance)
# We use this for video processing to keep IDs consistent
if model_loaded:
    tracker = TrafficTracker(model, device)
else:
    tracker = None

# --- HELPER: DRAWING FUNCTION ---
def draw_results(image, detections):
    """
    image: OpenCV BGR image
    detections: List of dicts {'bbox': [x1,y1,x2,y2], 'score': float, 'label': int, 'id': int (optional)}
    """
    draw_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        score = det.get('score', 0.0)
        track_id = det.get('id', None)
        
        # Color: Green for detection, Orange for tracked object
        color = (0, 255, 0) if track_id is None else (0, 165, 255)
        
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{score:.2f}"
        if track_id is not None:
            label_text = f"ID: {track_id}"
            
        cv2.putText(draw_img, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return draw_img

# --- HELPER: INFERENCE FOR SINGLE IMAGE ---
def process_single_frame(img_bgr):
    """
    Runs raw detection (No DeepSORT) for a single static image.
    """
    # 1. Prepare Image (RGB for model)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Inference
    with torch.no_grad():
        # Pass training=False to trigger feature extraction
        proposals, labels, scores, deltas = model(img_rgb, None, training=False)

    if len(proposals) == 0:
        return []

    # 3. Apply Regression & Clamping (Same logic as your notebook)
    dw = deltas[:, 2].clamp(min=-1.0, max=1.0)
    dh = deltas[:, 3].clamp(min=-1.0, max=1.0)
    
    p_w = proposals[:, 2] - proposals[:, 0]
    p_h = proposals[:, 3] - proposals[:, 1]
    p_ctr_x = proposals[:, 0] + 0.5 * p_w
    p_ctr_y = proposals[:, 1] + 0.5 * p_h
    
    pred_ctr_x = p_ctr_x + p_w * deltas[:, 0]
    pred_ctr_y = p_ctr_y + p_h * deltas[:, 1]
    pred_w = p_w * torch.exp(dw)
    pred_h = p_h * torch.exp(dh)
    
    refined_boxes = torch.stack([
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ], dim=1)

    # 4. Dynamic Scaling (Recover coordinates)
    # Note: We rely on the internal resize logic. 
    # If your notebook had explicit scaling, add it here.
    # For now, assuming model returns coordinates relative to input size.
    
    # 5. NMS
    keep = torchvision.ops.nms(refined_boxes, scores, iou_threshold=0.3)
    
    results = []
    final_boxes = refined_boxes[keep].cpu().numpy()
    final_scores = scores[keep].cpu().numpy()
    final_labels = labels[keep].cpu().numpy()
    
    for i in range(len(final_boxes)):
        if final_scores[i] > 0.5: # Threshold
            results.append({
                'bbox': final_boxes[i],
                'score': final_scores[i],
                'label': final_labels[i]
            })
            
    return results

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process_image', methods=['POST'])
def process_image():
    if not model_loaded: return jsonify({'error': 'Model not loaded'}), 500
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'}), 400

    try:
        # 1. Load Image as OpenCV BGR
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 2. Run Detection
        detections = process_single_frame(img_bgr)
        
        # 3. Draw & Save
        result_img = draw_results(img_bgr, detections)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        cv2.imwrite(result_path, result_img)
        
        return jsonify({'success': True, 'count': len(detections)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    if not model_loaded: return jsonify({'error': 'Model not loaded'}), 500
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return jsonify({'error': 'Bad video'}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_video.mp4')
    # Use 'avc1' or 'h264' for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Run Tracker
            # tracker.process_frame handles inference, NMS, and DeepSORT
            tracks = tracker.process_frame(frame)
            
            # 2. Convert Tracks to Drawing Format
            detections = []
            for t in tracks:
                detections.append({
                    'bbox': t['bbox'],
                    'id': t['id'],
                    'label': t['class_id']
                })
            
            # 3. Draw
            drawn_frame = draw_results(frame, detections)
            out.write(drawn_frame)

        cap.release()
        out.release()
        return jsonify({'success': True})
    except Exception as e:
        cap.release()
        out.release()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)