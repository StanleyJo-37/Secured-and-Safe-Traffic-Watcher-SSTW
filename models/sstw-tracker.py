from classical_sstw import ClassicalSSTWModel
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
import cv2

class SSTWTracker():
  def __init__(
    self,
    detector: torch.nn.Module=ClassicalSSTWModel
  ):
    self.detector = detector
    self.tracker = DeepSort()

    self.detector.eval()
    self.frames = []
    self.ids = []
    
  def convert_detections(detections, threshold, classes):
    # Get the bounding boxes, labels and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    # Filter out low confidence scores and non-person classes.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for i, box in enumerate(boxes):
      # Append ([x, y, w, h], score, label_string).
      final_boxes.append(
        (
          [box[0], box[1], box[2] - box[0], box[3] - box[1]],
          scores[i],
          str(labels[i])
        )
      )

    return final_boxes
    
  def annotate(tracks, frame, resized_frame, frame_width, frame_height, colors):
    for track in tracks:
      if not track.is_confirmed():
          continue
      track_id = track.track_id
      track_class = track.det_class
      x1, y1, x2, y2 = track.to_ltrb()
      p1 = (int(x1/resized_frame.shape[1]*frame_width), int(y1/resized_frame.shape[0]*frame_height))
      p2 = (int(x2/resized_frame.shape[1]*frame_width), int(y2/resized_frame.shape[0]*frame_height))
      # Annotate boxes.
      color = colors[int(track_class)]
      cv2.rectangle(
          frame,
          p1,
          p2,
          color=(int(color[0]), int(color[1]), int(color[2])),
          thickness=2
      )
      # Annotate ID.
      cv2.putText(
          frame, f"ID: {track_id}",
          (p1[0], p1[1] - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
          lineType=cv2.LINE_AA
      )
    return frame
  
  def apply_deltas(boxes, deltas):
    
    
  def detect_and_track(self, frames):
    with torch.no_grad():
      final_boxes, final_labels, final_scores, final_deltas = self.detector(frames.permute(1, 2, 0).cpu().numpy(), None, training=False)
    
    tracked_frames = []
    for frame in frames:
      refined_bboxes = self.apply_deltas(final_deltas)
      refined_bboxes = 
      self.tracker.update_tracks(refined_bboxes, )
    
  def track(self, )