from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision
import os
import cv2
import numpy as np
from classical_sstw import ClassicalSSTWModel

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
model = ClassicalSSTWModel(num_classes=7)
try:
    model.load_state_dict(torch.load('outputs/best_model_new.pth', map_location=device))
    model.to(device)
    model.eval()
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model_loaded = False
    model = None

def draw_results(image, detections):
    draw_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        score = det.get('score', 0.0)
        track_id = det.get('id', None)
        
        color = (0, 255, 0) if track_id is None else (0, 165, 255)
        
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{score:.2f}"
        if track_id is not None:
            label_text = f"ID: {track_id}"
            
        cv2.putText(draw_img, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return draw_img

def process_single_frame(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        proposals, labels, scores, deltas = model(img_rgb, None, training=False)

    if len(proposals) == 0:
        return []

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
        pred_ctr_x - 0.5 * pred_w * 4.,
        pred_ctr_y - 0.5 * pred_h * 4.,
        pred_ctr_x + 0.5 * pred_w * 4.,
        pred_ctr_y + 0.5 * pred_h * 4.
    ], dim=1)

    keep = torchvision.ops.nms(refined_boxes, scores, iou_threshold=0.3)
    
    results = []
    final_boxes = refined_boxes[keep].cpu().numpy()
    final_scores = scores[keep].cpu().numpy()
    final_labels = labels[keep].cpu().numpy()
    
    for i in range(len(final_boxes)):
        if final_scores[i] > 0.5:
            results.append({
                'bbox': final_boxes[i],
                'score': final_scores[i],
                'label': final_labels[i]
            })
            
    return results

# Store last analysis result globally
last_analysis = {
    "safety_score": 0,
    "traffic_density": "Low",
    "violations_detected": 0,
    "details": []
}

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process_image', methods=['POST'])
def process_image():
    global last_analysis
    if not model_loaded: return jsonify({'error': 'Model not loaded'}), 500
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        detections = process_single_frame(img_bgr)
        
        result_img = draw_results(img_bgr, detections)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        cv2.imwrite(result_path, result_img)
        
        # 4. Update analysis result
        num_detections = len(detections)
        if num_detections == 0:
            density = "Low"
            safety = 95
        elif num_detections <= 3:
            density = "Low"
            safety = 85
        elif num_detections <= 7:
            density = "Medium"
            safety = 70
        else:
            density = "High"
            safety = 50
        
        last_analysis = {
            "safety_score": safety,
            "traffic_density": density,
            "violations_detected": num_detections,
            "details": [
                {"timestamp": "N/A", "type": f"Detection {i+1}", "severity": "Medium"}
                for i in range(min(num_detections, 5))
            ]
        }
        
        return jsonify({'success': True, 'count': num_detections})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis_result', methods=['GET'])
def get_analysis_result():
    return jsonify(last_analysis)

@app.route('/result_image')
def get_result_image():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    return jsonify({'error': 'No result available'}), 404

# @app.route('/process_video', methods=['POST'])
# def process_video():
#     if not model_loaded: return jsonify({'error': 'Model not loaded'}), 500
#     file = request.files.get('file')
#     if not file: return jsonify({'error': 'No file'}), 400

#     input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
#     file.save(input_path)

#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened(): return jsonify({'error': 'Bad video'}), 500

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
#     output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_video.mp4')
#     # Use 'avc1' or 'h264' for better browser compatibility
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret: break

#             # 1. Run Tracker
#             # tracker.process_frame handles inference, NMS, and DeepSORT
#             tracks = tracker.process_frame(frame)
            
#             # 2. Convert Tracks to Drawing Format
#             detections = []
#             for t in tracks:
#                 detections.append({
#                     'bbox': t['bbox'],
#                     'id': t['id'],
#                     'label': t['class_id']
#                 })
            
#             # 3. Draw
#             drawn_frame = draw_results(frame, detections)
#             out.write(drawn_frame)

#         cap.release()
#         out.release()
#         return jsonify({'success': True})
#     except Exception as e:
#         cap.release()
#         out.release()
#         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)