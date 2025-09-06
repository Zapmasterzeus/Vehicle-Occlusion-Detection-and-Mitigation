# script-video-pipeline/advanced_occlusion_detector.py
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
import cv2
from typing import List, Dict, Tuple, Optional

class AdvancedOcclusionDetector:
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 appearance_threshold: float = 0.5,
                 motion_consistency_weight: float = 0.4,
                 depth_consistency_weight: float = 0.3,
                 appearance_weight: float = 0.3):
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)
        self.appearance_features = {}

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

    def detect_occlusions(self, detections: List[Dict]) -> List[Dict]:
        """Detect occlusions in the current frame."""
        if not detections:
            return detections

        # Update track history
        for det in detections:
            track_id = det['track_id']
            bbox = det['bbox']
            self.track_history[track_id].append({
                'bbox': bbox.copy(),
                'depth': det.get('depth', {}).get('mean', 0)
            })
            # Keep only last 10 frames of history
            self.track_history[track_id] = self.track_history[track_id][-10:]

        # Check for occlusions between all pairs of detections
        for i in range(len(detections)):
            det1 = detections[i]
            track_id1 = det1['track_id']
            bbox1 = det1['bbox']
            
            for j in range(i + 1, len(detections)):
                det2 = detections[j]
                track_id2 = det2['track_id']
                bbox2 = det2['bbox']
                
                # Calculate IOU
                iou = self.calculate_iou(bbox1, bbox2)
                if iou < self.iou_threshold:
                    continue
                
                # Simple occlusion decision based on depth
                depth1 = det1.get('depth', {}).get('mean', 0)
                depth2 = det2.get('depth', {}).get('mean', 0)
                
                if depth1 > depth2 + 0.5:  # Object 1 is further away
                    det1['occluded'] = True
                    det1['occluded_by'] = track_id2
                elif depth2 > depth1 + 0.5:  # Object 2 is further away
                    det2['occluded'] = True
                    det2['occluded_by'] = track_id1
                # If depths are similar, use track history
                else:
                    if len(self.track_history[track_id1]) > len(self.track_history[track_id2]):
                        det2['occluded'] = True
                        det2['occluded_by'] = track_id1
                    else:
                        det1['occluded'] = True
                        det1['occluded_by'] = track_id2
        
        return detections