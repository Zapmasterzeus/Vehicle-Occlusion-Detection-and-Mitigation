# script-video-pipeline/advanced_occlusion.py
import json
import os
import sys
import glob
from advanced_occlusion_detector import AdvancedOcclusionDetector

def process_frames(input_dir: str, output_dir: str, deepsort_json: str):
    """Process frames with occlusion detection and generate visualizations."""
    # Load DeepSORT detections
    with open(deepsort_json, 'r') as f:
        data = json.load(f)
    
    detector = AdvancedOcclusionDetector()
    
    # Create output directory for visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame
    for frame in data.get('frames', []):
        if 'objects' in frame and 'name' in frame:
            # Get corresponding image path
            img_path = os.path.join(input_dir, frame['name'])
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found")
                continue
                
            # Run occlusion detection
            frame_objects = frame['objects']
            processed_objects = detector.detect_occlusions(
                frame_objects,
                image_path=img_path,
                output_dir=output_dir
            )
            
            # Update frame with processed objects
            frame['objects'] = processed_objects
    
    # Save updated JSON
    output_json = os.path.join(output_dir, 'aod_output.json')
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Occlusion detection complete. Results saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python advanced_occlusion.py <input_dir> <output_dir> <deepsort_json>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    deepsort_json = sys.argv[3]
    
    if not os.path.exists(deepsort_json):
        print(f"Error: File {deepsort_json} does not exist")
        sys.exit(1)
    
    process_frames(input_dir, output_dir, deepsort_json)