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

    def poly2d_to_bbox(self, poly2d):
        """Convert polygon to bounding box [x1, y1, x2, y2]."""
        points = np.array([[p[0], p[1]] for p in poly2d if len(p) >= 2])
        if len(points) == 0:
            return [0, 0, 0, 0]
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        return [float(x1), float(y1), float(x2), float(y2)]
        
    def poly_area(self, x, y):
        """Calculate the area of a polygon using Shoelace formula."""
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
    def poly_intersection_area(self, poly1, poly2):
        """Calculate intersection area between two polygons."""
        # Convert to Shapely polygons for intersection calculation
        from shapely.geometry import Polygon
        p1 = Polygon([(p[0], p[1]) for p in poly1])
        p2 = Polygon([(p[0], p[1]) for p in poly2])
        
        if not p1.is_valid or not p2.is_valid:
            return 0.0
            
        intersection = p1.intersection(p2)
        return intersection.area if not intersection.is_empty else 0.0
        
    def calculate_iou(self, poly1, poly2):
        """Calculate Intersection over Union between two polygons."""
        # Calculate intersection area
        intersection = self.poly_intersection_area(poly1, poly2)
        
        # Calculate areas of both polygons
        x1, y1 = zip(*[(p[0], p[1]) for p in poly1])
        area1 = self.poly_area(x1, y1)
        
        x2, y2 = zip(*[(p[0], p[1]) for p in poly2])
        area2 = self.poly_area(x2, y2)
        
        # Calculate union
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

    def draw_occlusion_visualization(self, image_path: str, detections: List[Dict], output_path: str) -> None:
        """Draw visualization of occlusions on the image.
        
        Args:
            image_path: Path to the input image
            detections: List of detection objects with occlusion info
            output_path: Path to save the visualization
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return
            
        # Group occluders and occluded objects
        occluders = {}
        occluded = {}
        
        for det in detections:
            if 'poly2d' not in det:
                continue
                
            track_id = det.get('track_id')
            if track_id is None:
                continue
                
            if det.get('occluded', False):
                occluded[track_id] = det
            else:
                occluders[track_id] = det
        
        # Draw occluded objects first (red)
        for track_id, det in occluded.items():
            try:
                # Convert poly2d to points for drawing
                points = np.array([(int(p[0]), int(p[1])) for p in det['poly2d']], np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Draw filled polygon
                cv2.fillPoly(img, [points], (0, 0, 255))  # Red for occluded
                
                # Draw track ID
                bbox = self.poly2d_to_bbox(det['poly2d'])
                cv2.putText(img, f"ID:{track_id}", (int(bbox[0]), int(bbox[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing occluded object {track_id}: {e}")
        
        # Draw occluding objects (blue)
        for track_id, det in occluders.items():
            try:
                # Convert poly2d to points for drawing
                points = np.array([(int(p[0]), int(p[1])) for p in det['poly2d']], np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Draw filled polygon
                cv2.fillPoly(img, [points], (255, 0, 0))  # Blue for occluders
                
                # Draw track ID
                bbox = self.poly2d_to_bbox(det['poly2d'])
                cv2.putText(img, f"ID:{track_id}", (int(bbox[0]), int(bbox[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing occluding object {track_id}: {e}")
        
        # Save the visualization
        cv2.imwrite(output_path, img)

    def detect_occlusions(self, detections: List[Dict], image_path: str = None, output_dir: str = None) -> List[Dict]:
        """Detect occlusions in the current frame.
        
        Args:
            detections: List of detection objects
            image_path: Optional path to the input image for visualization
            output_dir: Optional directory to save visualization
            
        Returns:
            List of detections with occlusion information
        """
        if not detections:
            return detections

        # Update track history and ensure all detections have required fields
        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue
                
            # Ensure we have poly2d data
            if 'poly2d' not in det and 'bbox' in det:
                # If we only have bbox, create a simple rectangular poly2d
                x1, y1, x2, y2 = det['bbox']
                det['poly2d'] = [
                    [x1, y1, 0],
                    [x2, y1, 0],
                    [x2, y2, 0],
                    [x1, y2, 0]
                ]
            
            # Initialize track history if needed
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                
            # Update track history with current state
            self.track_history[track_id].append({
                'poly2d': det.get('poly2d', []),
                'depth': det.get('depth', {}).get('mean', 0),
                'timestamp': len(self.track_history[track_id])
            })
            # Keep only last 10 frames of history
            self.track_history[track_id] = self.track_history[track_id][-10:]
            
            # Initialize occlusion status
            det['occluded'] = False
            det['occluded_by'] = -1

        # Check for occlusions between all pairs of detections
        for i in range(len(detections)):
            det1 = detections[i]
            if 'poly2d' not in det1:
                continue
                
            track_id1 = det1.get('track_id')
            poly1 = det1['poly2d']
            
            for j in range(i + 1, len(detections)):
                det2 = detections[j]
                if 'poly2d' not in det2:
                    continue
                    
                track_id2 = det2.get('track_id')
                if track_id1 == track_id2:
                    continue  # Skip same track
                    
                poly2 = det2['poly2d']
                
                # Calculate IOU using polygons
                try:
                    iou = self.calculate_iou(poly1, poly2)
                    if iou < self.iou_threshold:
                        continue
                        
                    # Get depth information
                    depth1 = det1.get('depth', {}).get('mean', 0)
                    depth2 = det2.get('depth', {}).get('mean', 0)
                    
                    # Check if one object is significantly behind the other
                    if depth1 > depth2 + 0.5:  # Object 1 is further away
                        det1['occluded'] = True
                        det1['occluded_by'] = track_id2
                    elif depth2 > depth1 + 0.5:  # Object 2 is further away
                        det2['occluded'] = True
                        det2['occluded_by'] = track_id1
                        
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error calculating IOU between {track_id1} and {track_id2}: {str(e)}")
                    continue
                # If depths are similar, use track history
                else:
                    if len(self.track_history[track_id1]) > len(self.track_history[track_id2]):
                        det2['occluded'] = True
                        det2['occluded_by'] = track_id1
                    else:
                        det1['occluded'] = True
                        det1['occluded_by'] = track_id2
        
        # Save visualization if image path and output directory are provided
        if image_path and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                frame_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{frame_name}")
                self.draw_occlusion_visualization(image_path, detections, output_path)
            except Exception as e:
                print(f"Error saving occlusion visualization: {e}")
                
        return detections