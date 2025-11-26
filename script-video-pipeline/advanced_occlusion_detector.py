# script-video-pipeline/advanced_occlusion_detector.py
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
import cv2
import os
from typing import List, Dict, Tuple, Optional

class AdvancedOcclusionDetector:
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 appearance_threshold: float = 0.5,
                 motion_consistency_weight: float = 0.4,
                 depth_consistency_weight: float = 0.3,
                 appearance_weight: float = 0.3,
                 depth_delta: float = 0.5,
                 partial_thresh: float = 0.35,
                 major_thresh: float = 0.7,
                 no_occ_thresh: float = 0.05):
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)
        self.appearance_features = {}
        self.depth_delta = depth_delta
        self.partial_thresh = partial_thresh
        self.major_thresh = major_thresh
        self.no_occ_thresh = no_occ_thresh

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
        """Calculate intersection area between two polygons. Falls back to bbox intersection if shapely is unavailable."""
        try:
            from shapely.geometry import Polygon
            p1 = Polygon([(p[0], p[1]) for p in poly1])
            p2 = Polygon([(p[0], p[1]) for p in poly2])
            if not p1.is_valid or not p2.is_valid:
                return 0.0
            intersection = p1.intersection(p2)
            return intersection.area if not intersection.is_empty else 0.0
        except Exception:
            # Fallback to bbox intersection area
            b1 = self.poly2d_to_bbox(poly1)
            b2 = self.poly2d_to_bbox(poly2)
            xA = max(b1[0], b2[0])
            yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2])
            yB = min(b1[3], b2[3])
            interW = max(0.0, xB - xA)
            interH = max(0.0, yB - yA)
            return interW * interH
        
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
        """Draw visualization of occlusions on the image using severity colors.
        Colors:
          blue(b)=no occlusion, yellow(y)=partial, orange(o)=major, red(r)=full
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return

        color_map = {
            'b': (255, 0, 0),      # blue
            'y': (0, 255, 255),    # yellow
            'o': (0, 165, 255),    # orange
            'r': (0, 0, 255),      # red
        }

        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            cat = det.get('occlusion_category', 'b')
            color = color_map.get(cat, (255, 0, 0))
            try:
                poly = det.get('poly2d_adjusted') if det.get('poly2d_adjusted') else det.get('poly2d')
                if not poly:
                    continue
                points = np.array([(int(p[0]), int(p[1])) for p in poly], np.int32)
                if points.size == 0:
                    continue
                points = points.reshape((-1, 1, 2))
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], color)
                img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

                bbox = self.poly2d_to_bbox(poly)
                cv2.putText(img, f"ID:{track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing object {track_id}: {e}")

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

        # Initialize per-object occlusion summary
        n = len(detections)
        poly_list = []
        area_list = []
        for det in detections:
            # Prefer adjusted polygon if provided by DeepSORT step
            if 'poly2d_adjusted' in det and det['poly2d_adjusted']:
                eff_poly = det['poly2d_adjusted']
            elif 'poly2d' in det and det['poly2d']:
                eff_poly = det['poly2d']
            elif 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                eff_poly = [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0]]
            else:
                eff_poly = []
            poly_list.append(eff_poly)
            if eff_poly:
                xs, ys = zip(*[(p[0], p[1]) for p in eff_poly])
                area_list.append(self.poly_area(np.array(xs), np.array(ys)))
            else:
                area_list.append(0.0)


        occl_ratio = [0.0] * n
        occl_by = [-1] * n

        # Check pairs and accumulate max occlusion ratio per object (using effective polygons)
        for i in range(n):
            if not poly_list[i] or detections[i].get('track_id') is None:
                continue
            for j in range(i + 1, n):
                if not poly_list[j] or detections[j].get('track_id') is None:
                    continue
                track_id1 = detections[i].get('track_id')
                track_id2 = detections[j].get('track_id')
                if track_id1 == track_id2:
                    continue
                try:
                    iou_val = self.calculate_iou(poly_list[i], poly_list[j])
                    if iou_val < self.iou_threshold:
                        continue
                    inter = self.poly_intersection_area(poly_list[i], poly_list[j])
                    r_i = inter / (area_list[i] + 1e-6)
                    r_j = inter / (area_list[j] + 1e-6)

                    d1 = detections[i].get('depth', {}).get('mean', 0)
                    d2 = detections[j].get('depth', {}).get('mean', 0)

                    # Decide who is behind
                    behind_i = False
                    behind_j = False
                    if d1 > d2 + self.depth_delta:
                        behind_i = True
                    elif d2 > d1 + self.depth_delta:
                        behind_j = True
                    else:
                        # Similar depth: shorter history considered behind
                        h1 = len(self.track_history.get(track_id1, []))
                        h2 = len(self.track_history.get(track_id2, []))
                        if h1 > h2:
                            behind_j = True
                        else:
                            behind_i = True

                    if behind_i and r_i > occl_ratio[i]:
                        occl_ratio[i] = r_i
                        occl_by[i] = track_id2
                    if behind_j and r_j > occl_ratio[j]:
                        occl_ratio[j] = r_j
                        occl_by[j] = track_id1
                except Exception as e:
                    print(f"Error calculating occlusion between {track_id1} and {track_id2}: {str(e)}")
                    continue

        # Assign final flags and categories
        def to_category(r: float) -> str:
            if r <= self.no_occ_thresh:
                return 'b'  # no occlusion
            if r < self.partial_thresh:
                return 'y'  # partial
            if r < self.major_thresh:
                return 'o'  # major
            return 'r'      # full

        for idx, det in enumerate(detections):
            cat = to_category(occl_ratio[idx])
            det['occlusion_category'] = cat
            det['occlusion_score'] = float(occl_ratio[idx])
            if cat == 'b':
                det['occluded'] = False
                det['occluded_by'] = -1
            else:
                det['occluded'] = True
                det['occluded_by'] = occl_by[idx]
        
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