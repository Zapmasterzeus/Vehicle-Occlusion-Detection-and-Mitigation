# script-video-pipeline/advanced_occlusion.py
import json
import os
import sys
from advanced_occlusion_detector import AdvancedOcclusionDetector

def process_json(json_path: str):
    """Process a single JSON file with detections."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    detector = AdvancedOcclusionDetector()
    
    # Process each frame
    for frame in data.get('frames',[]):
        if 'objects' in frame:
            frame['objects'] = detector.detect_occlusions(frame['objects'])
    
    # Save back to the same file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python advanced_occlusion.py <path_to_deepsort_output.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist")
        sys.exit(1)
    
    process_json(json_path)