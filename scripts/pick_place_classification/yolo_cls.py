from ultralytics import YOLO
import os
import shutil

train = False

if train:
    model = YOLO("yolov8m-cls.pt")  
    save_dir = "./../models/runs"

    # Train classifier
    results = model.train(
        data="./../data/lego-pickplace-5",  # dataset folder name (Roboflow export gives this)
        imgsz=640,                          # standard image size for classifiers
        batch=32,
        epochs=100,
        plots=True,
        save_dir=save_dir
    )
else:
    model = YOLO("./../models/pick_place_yolo_082125.pt")  

# Input + output dirs
input_dir = "./../data/pickplace_results/test"
output_dir = "./../data/pickplace_results/test_sorted"
os.makedirs(output_dir, exist_ok=True)

# Run predictions
results = model.predict(source=input_dir)
print('moving files...')

for r in results:
    # Predicted class name
    pred_class = r.names[r.probs.top1]
    
    # Confidence of top prediction
    conf = r.probs.top1conf  # e.g., 0.87
    
    # Ensure class folder exists
    class_dir = os.path.join(output_dir, pred_class)
    os.makedirs(class_dir, exist_ok=True)
    
    # Original filename
    base_name = os.path.basename(r.path)
    name, ext = os.path.splitext(base_name)
    
    # New filename with confidence (2 decimal places)
    new_name = f"{conf:.2f}_{name}{ext}"
    
    # Copy image into predicted class folder with new name
    shutil.copy(r.path, os.path.join(class_dir, new_name))
    
    # Optional debug
    # print(f"Copied {r.path} → {os.path.join(class_dir, new_name)}")