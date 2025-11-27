
from ultralytics import YOLO
import os

if __name__ == '__main__':

    print("[INFO] Loading YOLOv8-Classification model...")
    model = YOLO('yolov8n-cls.pt') 


    print("[INFO] Starting training...")
    results = model.train(
        data='./',         
        epochs=50, 
        imgsz=224, 
        batch=16,          
        project='seagrass_yolo',
        name='run1'
    )

    print("[INFO] Training finished!")
    print(f"[INFO] Best model saved at: {results.save_dir}/weights/best.pt")