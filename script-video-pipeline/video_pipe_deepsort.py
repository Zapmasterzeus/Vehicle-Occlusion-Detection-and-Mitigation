import os
import sys
import json
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

def poly2d_to_bbox(poly2d):
    points = np.array([[p[0], p[1]] for p in poly2d if len(p) >= 2])
    if points.shape[0] == 0:
        return [0, 0, 0, 0]
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]

import cv2

def draw_bbox_and_id(img, bbox, track_id, category, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {category}"
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

class EnhancedDeepSORT:
    def __init__(self, max_age=60, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3):
        """Initialize the enhanced DeepSORT tracker with improved parameters."""
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # Store last 30 positions
        self.class_names = [
            "person", "bike", "car", "motor", "bus", "train", "truck", "traffic light", "traffic sign", "rider"
        ]
        self.all_categories = sorted(self.class_names)
        self.category_to_id = {cat: i for i, cat in enumerate(self.all_categories)}
        self.next_track_id = 0

    def update(self, detections, frame):
        """Update tracker with new detections and frame."""
        # Convert detections to DeepSORT format
        bbs = []
        confs = []
        class_ids = []
        
        for det in detections:
            bbox = poly2d_to_bbox(det['poly2d']) if 'poly2d' in det else det['bbox']
            confidence = float(det.get('confidence', 1.0))
            class_id = self.category_to_id.get(det.get('category', 'car'), 0)
            # Attach id to det for later propagation
            det_id = det.get('id')
            bbs.append(bbox)
            confs.append(confidence)
            class_ids.append(class_id)
        
        # Update tracker
        tracks = self.tracker.update_tracks(
            list(zip(bbs, confs, class_ids)),
            frame=frame
        )
        
        # Process tracks
        results = []
        for i, track in enumerate(tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = track.get_det_class()
            # Propagate id from detections to results (1:1 order assumption)
            det_id = detections[i].get('id') if i < len(detections) else None
            # Update track history
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[track_id].append(center)
            results.append({
                'id': det_id,
                'track_id': track_id,
                'bbox': bbox.tolist(),
                'class_id': class_id,
                'category': self.all_categories[class_id],
                'confidence': track.confidence if hasattr(track, 'confidence') else 1.0,
                'age': track.age,
                'time_since_update': track.time_since_update
            })
            
        return results

def process_frames(input_dir, output_dir, midas_json_path):
    # Load depth data
    with open(midas_json_path, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    
    # Initialize enhanced tracker
    tracker = EnhancedDeepSORT(
        max_age=60,               # Maximum frames to keep a track alive without updates
        n_init=3,                 # Number of consecutive detections before confirming track
        max_iou_distance=0.7,     # Maximum IoU distance for matching
        max_cosine_distance=0.3   # Maximum cosine distance for appearance matching
    )
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted frame files
    frame_files = sorted([f for f in os.listdir(input_dir) 
                         if f.startswith('frame_') and f.endswith('.jpg')])
    
    # Process each frame
    for frame_idx, (frame, frame_file) in enumerate(zip(frames, frame_files)):
        if 'objects' not in frame:
            continue
            
        # Prepare detections for tracking
        detections = []
        for obj in frame['objects']:
            if 'poly2d' in obj:
                bbox = poly2d_to_bbox(obj['poly2d'])
            elif 'bbox' in obj:
                bbox = obj['bbox']
            else:
                continue
                
            det = {
                'bbox': bbox,
                'confidence': float(obj.get('confidence', 1.0)),
                'category': obj.get('category', 'car'),
                'depth': obj.get('depth', {'mean': 0})
            }
            detections.append(det)
        
        # Read frame image
        img_path = os.path.join(input_dir, frame_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Update tracker with current frame
        tracked_objects = tracker.update(detections, img)
        
        # Update frame objects with tracking information
        output_objects = []
        for obj in tracked_objects:
            output_obj = {
                'id': obj.get('id'),
                'track_id': obj['track_id'],
                'occluded': False,
                'occluded_by': -1
            }
            output_objects.append(output_obj)
        
        # Update frame objects with tracking information
        for obj in output_objects:
            # Find the original object that matches this track_id
            for orig_obj in frame['objects']:
                if orig_obj.get('id') == obj['id']:
                    # Update the original object with tracking info
                    orig_obj.update({
                        'track_id': obj.get('track_id'),
                        'occluded': obj.get('occluded', False),
                        'occluded_by': obj.get('occluded_by', -1)
                    })
                    break
        
        # Draw visualization (optional)
        if output_dir:
            vis_img = img.copy()
            for obj in frame['objects']:
                if 'track_id' not in obj:
                    continue
                    
                # Get bbox from poly2d if available, otherwise use bbox
                if 'poly2d' in obj and obj['poly2d']:
                    bbox = poly2d_to_bbox(obj['poly2d'])
                elif 'bbox' in obj:
                    bbox = obj['bbox']
                else:
                    continue
                    
                color = (0, 255, 0)  # Default to green (not occluded)
                
                # Draw the bounding box and ID
                vis_img = draw_bbox_and_id(
                    vis_img, 
                    bbox,
                    obj['track_id'],
                    obj.get('category', 'object'),
                    color=color
                )
            
            # Save visualization
            output_path = os.path.join(output_dir, f"{frame_file}")
            cv2.imwrite(output_path, vis_img)
    
    # Save updated JSON with tracking information
    output_json_path = os.path.join(os.path.dirname(midas_json_path), 'deepsort_output.json')
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_json_path



if __name__ == "__main__":
    if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        midas_json_path = sys.argv[3]
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory.")
            sys.exit(1)
        os.makedirs(output_dir, exist_ok=True)
        process_frames(input_dir, output_dir, midas_json_path)
    else:
        print("Usage: python video_pipe_deepsort.py <input_dir> <output_dir> <midas_json_path>")
        sys.exit(1)
