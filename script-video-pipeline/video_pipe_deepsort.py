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
        
        # Helper to compute IoU between two [x1, y1, x2, y2] boxes
        def _iou(boxA, boxB):
            xA = max(float(boxA[0]), float(boxB[0]))
            yA = max(float(boxA[1]), float(boxB[1]))
            xB = min(float(boxA[2]), float(boxB[2]))
            yB = min(float(boxA[3]), float(boxB[3]))
            interW = max(0.0, xB - xA)
            interH = max(0.0, yB - yA)
            interArea = interW * interH
            boxAArea = max(0.0, (float(boxA[2]) - float(boxA[0]))) * max(0.0, (float(boxA[3]) - float(boxA[1])))
            boxBArea = max(0.0, (float(boxB[2]) - float(boxB[0]))) * max(0.0, (float(boxB[3]) - float(boxB[1])))
            denom = (boxAArea + boxBArea - interArea)
            return interArea / (denom + 1e-6)

        # Process tracks
        results = []
        for track in tracks:
            if track.time_since_update > 1:
                continue
                
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = track.get_det_class()

            # Find best matching detection by IoU to propagate original detection ID and category
            det_id = None
            det_category = None
            best_iou = 0.0
            for det in detections:
                det_bbox = det.get('bbox') if 'bbox' in det else poly2d_to_bbox(det.get('poly2d', []))
                iou_val = _iou(bbox, det_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    det_id = det.get('id')
                    det_category = det.get('category', 'car')

            # Update track history
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history[track_id].append(center)
            results.append({
                'id': det_id,
                'track_id': track_id,
                'bbox': bbox.tolist(),
                'class_id': class_id if class_id is not None else -1,
                'category': det_category if det_category is not None else (self.all_categories[class_id] if class_id is not None and 0 <= class_id < len(self.all_categories) else 'object'),
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
    
    def _iou_boxes(boxA, boxB):
        xA = max(float(boxA[0]), float(boxB[0]))
        yA = max(float(boxA[1]), float(boxB[1]))
        xB = min(float(boxA[2]), float(boxB[2]))
        yB = min(float(boxA[3]), float(boxB[3]))
        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0.0, (float(boxA[2]) - float(boxA[0]))) * max(0.0, (float(boxA[3]) - float(boxA[1])))
        boxBArea = max(0.0, (float(boxB[2]) - float(boxB[0]))) * max(0.0, (float(boxB[3]) - float(boxB[1])))
        denom = (boxAArea + boxBArea - interArea)
        return interArea / (denom + 1e-6)

    prev_objects = []
    # Maintain previous state per track id for area/depth/motion based adjustment
    prev_state_by_tid: Dict[int, Dict] = {}
    motion_hist_by_tid: Dict[int, deque] = {}
    next_fallback_track_id = 1000000
    
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
                'id': obj.get('id'),
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
        
        # Update and propagate tracking info to all objects (IoU-based only)
        track_bboxes = []
        for obj in tracked_objects:
            track_bboxes.append((obj['bbox'], obj['track_id']))

        for orig_obj in frame['objects']:
            if 'poly2d' in orig_obj and orig_obj['poly2d']:
                obox = poly2d_to_bbox(orig_obj['poly2d'])
            elif 'bbox' in orig_obj:
                obox = orig_obj['bbox']
            else:
                continue

            assigned_tid = None
            best_iou = 0.0
            best_tid = None
            for tbbox, tid in track_bboxes:
                iou_val = _iou_boxes(obox, tbbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid
            if best_tid is not None and best_iou > 0.05:
                assigned_tid = best_tid

            if assigned_tid is None and prev_objects:
                best_iou = 0.0
                best_tid = None
                for prev in prev_objects:
                    iou_val = _iou_boxes(obox, prev['bbox'])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_tid = prev['track_id']
                if best_tid is not None and best_iou > 0.1:
                    assigned_tid = best_tid

            if assigned_tid is None:
                assigned_tid = next_fallback_track_id
                next_fallback_track_id += 1

            orig_obj.update({
                'track_id': assigned_tid,
                'occluded': False,
                'occluded_by': -1
            })

        # Compute adjusted polygons using previous state and depth ratio
        # Helpers
        def _centroid_from_bbox(b):
            return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)

        # Read current frame image size for clamping
        img_path = os.path.join(input_dir, frame_file)
        img_h = img_w = None
        try:
            _img = cv2.imread(img_path)
            if _img is not None:
                img_h, img_w = _img.shape[:2]
        except Exception:
            pass

        for orig_obj in frame['objects']:
            tid = orig_obj.get('track_id')
            if tid is None:
                continue
            # Current geometry
            if 'poly2d' in orig_obj and orig_obj['poly2d']:
                cur_poly = orig_obj['poly2d']
                cur_bbox = poly2d_to_bbox(cur_poly)
            elif 'bbox' in orig_obj:
                cur_bbox = orig_obj['bbox']
                x1, y1, x2, y2 = cur_bbox
                cur_poly = [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0]]
            else:
                continue

            # Areas
            try:
                xs, ys = zip(*[(p[0], p[1]) for p in cur_poly])
                cur_area_poly = 0.5 * abs(np.dot(np.array(xs), np.roll(np.array(ys), 1)) - np.dot(np.array(ys), np.roll(np.array(xs), 1)))
            except Exception:
                cur_area_poly = max(0.0, (float(cur_bbox[2]) - float(cur_bbox[0]))) * max(0.0, (float(cur_bbox[3]) - float(cur_bbox[1])))
            cur_w = max(1.0, float(cur_bbox[2]) - float(cur_bbox[0]))
            cur_h = max(1.0, float(cur_bbox[3]) - float(cur_bbox[1]))
            cur_area_bbox = cur_w * cur_h

            # Depth ratio (MiDaS inverse depth)
            new_depth = float(orig_obj.get('depth', {}).get('mean', 0.0))
            prev_state = prev_state_by_tid.get(tid)
            if prev_state is None:
                # No adjustment for first observation
                orig_obj['poly2d_adjusted'] = cur_poly
                orig_obj['expected_area'] = cur_area_poly
                orig_obj['depth_ratio'] = 1.0
            else:
                old_depth = float(prev_state.get('depth', 0.0))
                depth_ratio = new_depth / (old_depth + 1e-6)
                depth_ratio = max(0.5, min(depth_ratio, 2.0))
                prev_poly = prev_state.get('poly2d')
                if not prev_poly:
                    prev_bbox = prev_state.get('bbox', cur_bbox)
                    x1, y1, x2, y2 = prev_bbox
                    prev_poly = [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0]]
                try:
                    pxs, pys = zip(*[(p[0], p[1]) for p in prev_poly])
                    prev_area_poly = 0.5 * abs(np.dot(np.array(pxs), np.roll(np.array(pys), 1)) - np.dot(np.array(pys), np.roll(np.array(pxs), 1)))
                except Exception:
                    pb = prev_state.get('bbox', cur_bbox)
                    prev_area_poly = max(0.0, (float(pb[2]) - float(pb[0]))) * max(0.0, (float(pb[3]) - float(pb[1])))

                expected_area = max(1.0, prev_area_poly * depth_ratio)

                # Motion estimation via centroid shift
                cx, cy = _centroid_from_bbox(cur_bbox)
                pb = prev_state.get('bbox', cur_bbox)
                pcx, pcy = _centroid_from_bbox(pb)
                dx_raw = cx - pcx
                dy_raw = cy - pcy
                dq = motion_hist_by_tid.get(tid)
                if dq and len(dq) > 0:
                    mx = sum([p[0] for p in dq]) / len(dq)
                    my = sum([p[1] for p in dq]) / len(dq)
                    dx = cx - mx
                    dy = cy - my
                else:
                    dx = dx_raw
                    dy = dy_raw

                # If current visible area is below expected (allow 5% tolerance), expand polygon along motion direction
                tolerance = 0.95
                # Ensure poly2d has numeric xy
                def _poly_xy(poly):
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    return xs, ys
                def _area_xy(xs, ys):
                    x = np.array(xs)
                    y = np.array(ys)
                    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                def _to_polyL(xs, ys):
                    return [[float(x), float(y), "L"] for x, y in zip(xs, ys)]
                poly2d_adj = cur_poly
                if cur_area_poly + 1e-6 < expected_area * tolerance and (abs(dx) > 1e-3 or abs(dy) > 1e-3):
                    # Unit motion vector
                    norm = (dx**2 + dy**2) ** 0.5
                    ux, uy = dx / norm, dy / norm
                    xs0, ys0 = _poly_xy(cur_poly)
                    # Binary search factor 'a' for forward-only expansion p' = p + a*max(0, (p-c)·u)*u
                    def _apply(a):
                        xs = []
                        ys = []
                        for x, y in zip(xs0, ys0):
                            s = (x - cx) * ux + (y - cy) * uy
                            shift = a * s if s > 0 else 0.0
                            xn = x + shift * ux
                            yn = y + shift * uy
                            if img_w is not None and img_h is not None:
                                xn = max(0.0, min(xn, img_w - 1.0))
                                yn = max(0.0, min(yn, img_h - 1.0))
                            xs.append(xn)
                            ys.append(yn)
                        return xs, ys, _area_xy(xs, ys)
                    target = expected_area
                    xs_best, ys_best = xs0, ys0
                    # Find upper bound
                    a_lo, a_hi = 0.0, 1.0
                    _, _, area_hi = _apply(a_hi)
                    tries = 0
                    while area_hi < target and a_hi < 10.0 and tries < 10:
                        a_hi *= 2.0
                        _, _, area_hi = _apply(a_hi)
                        tries += 1
                    # Binary search
                    for _ in range(12):
                        a_mid = 0.5 * (a_lo + a_hi)
                        xs_mid, ys_mid, area_mid = _apply(a_mid)
                        if area_mid >= target:
                            a_hi = a_mid
                            xs_best, ys_best = xs_mid, ys_mid
                        else:
                            a_lo = a_mid
                    poly2d_adj = _to_polyL(xs_best, ys_best)
                else:
                    # Keep original, ensure 'L' format
                    xs0, ys0 = _poly_xy(cur_poly)
                    poly2d_adj = _to_polyL(xs0, ys0)
                orig_obj['poly2d_adjusted'] = poly2d_adj
                orig_obj['expected_area'] = expected_area
                orig_obj['depth_ratio'] = depth_ratio

        # Prepare previous frame states for next iteration
        prev_objects = []
        for orig_obj in frame['objects']:
            if 'poly2d' in orig_obj and orig_obj['poly2d']:
                obox = poly2d_to_bbox(orig_obj['poly2d'])
            elif 'bbox' in orig_obj:
                obox = orig_obj['bbox']
            else:
                continue
            prev_objects.append({'bbox': obox, 'track_id': orig_obj.get('track_id')})
            # Update per-track previous state using adjusted polygon if available
            tid = orig_obj.get('track_id')
            if tid is not None:
                if 'poly2d_adjusted' in orig_obj and orig_obj['poly2d_adjusted']:
                    pb = poly2d_to_bbox(orig_obj['poly2d_adjusted'])
                    prev_state_by_tid[tid] = {
                        'poly2d': orig_obj['poly2d_adjusted'],
                        'bbox': pb,
                        'depth': float(orig_obj.get('depth', {}).get('mean', 0.0))
                    }
                else:
                    prev_state_by_tid[tid] = {
                        'poly2d': orig_obj.get('poly2d', None),
                        'bbox': obox,
                        'depth': float(orig_obj.get('depth', {}).get('mean', 0.0))
                    }
                if tid not in motion_hist_by_tid:
                    motion_hist_by_tid[tid] = deque(maxlen=5)
                cb = prev_state_by_tid[tid]['bbox']
                motion_hist_by_tid[tid].append(((float(cb[0]) + float(cb[2])) / 2.0, (float(cb[1]) + float(cb[3])) / 2.0))
        
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
                if 'poly2d_adjusted' in obj and obj['poly2d_adjusted']:
                    pts = np.array([(int(p[0]), int(p[1])) for p in obj['poly2d_adjusted']], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], True, (255, 0, 255), 2)
            
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
