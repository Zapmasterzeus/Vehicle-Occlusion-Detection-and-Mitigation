import os
import json
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import numpy as np

DEFAULT_ATTRIBUTES = {
    "occluded": False,
    "truncated": False,
    "trafficLightColor": "none"
}

def draw_minimal_bbox(img, x1, y1, x2, y2, class_name, color=(0, 255, 0)):
    """Draw a minimal bounding box with class name"""
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
    
    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    cv2.putText(img, class_name, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return img

def generate_json_for_image(image_path):
    output_dir = "outputs/yolo/json"
    img_input_dir = "img_input"
    img_output_dir = "outputs/yolo/img"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_input_dir, exist_ok=True)
    os.makedirs(img_output_dir, exist_ok=True)
    
    model = YOLO('runs/detect/train7/weights/best.pt')
    
    import shutil
    try:
        shutil.copy2(image_path, img_input_dir)
    except Exception as e:
        print(f"Failed to copy {image_path} to {img_input_dir}: {e}")
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    results = model(image_path)
    result = results[0]
    
    objects = []
    for idx, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        img = draw_minimal_bbox(img, x1, y1, x2, y2, class_name)
        obj = {
            "category": class_name,
            "id": idx,
            "attributes": DEFAULT_ATTRIBUTES.copy(),
            "box2d": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        objects.append(obj)
    
    img_output_path = os.path.join(img_output_dir, f"{image_name}.jpg")
    cv2.imwrite(img_output_path, img)
    print(f"Saved annotated image: {img_output_path}")
    
    json_data = {
        "name": image_name,
        "frames": [
            {
                "timestamp": 10000,
                "objects": objects
            }
        ],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "dawn/dusk"
        }
    }
    
    json_output_path = os.path.join(output_dir, f"{image_name}.json")
    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved JSON: {json_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_json_for_image(sys.argv[1])
    else:
        import tkinter as tk
        from tkinter import filedialog
        def browse_and_generate_json():
            root = tk.Tk()
            root.withdraw()
            image_paths = filedialog.askopenfilenames(
                title="Select image(s) to run YOLO and generate JSON",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png"),
                    ("All files", "*.*")
                ],
                initialdir="E:/Vehicle Occlusion/bdd_images/test"
            )
            if not image_paths:
                print("No images selected. Exiting...")
                return
            for image_path in image_paths:
                generate_json_for_image(image_path)
        browse_and_generate_json()
