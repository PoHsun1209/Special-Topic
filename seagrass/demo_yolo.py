# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="DeepSeagrass YOLO Demo")
    parser.add_argument("--mode", type=str, default="webcam", choices=["image", "video", "webcam"], help="Mode: image, video, or webcam")
    parser.add_argument("--path", type=str, help="Path to image or video")
    # Path to your trained model
    parser.add_argument("--model", type=str, default="seagrass_yolo/run1/weights/best.pt", help="Path to trained .pt file")
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Load Model
    print(f"[INFO] Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at {args.model}")
        print("Please run train_yolo.py first.")
        return

    model = YOLO(args.model)

    # 2. Define processing function
    def process_frame(frame):
        # Inference
        results = model(frame, verbose=False)
        
        # Get prediction
        top1_index = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        class_name = results[0].names[top1_index]

        # Draw
        display_frame = frame.copy()
        text = f"{class_name}: {confidence*100:.1f}%"
        
        color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
        cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        return display_frame

    # 3. Execution Modes
    if args.mode == "image":
        if not args.path:
            print("[ERROR] Please provide --path for image mode.")
            return
        
        # Windows path fix for cv2.imread with non-ascii characters
        # cv2.imread doesn't handle Chinese paths well in Windows sometimes
        # We use a numpy workaround if needed, but standard read first
        frame = cv2.imread(args.path)
        
        if frame is None:
            # Try numpy method for Chinese paths
            import numpy as np
            frame = cv2.imdecode(np.fromfile(args.path, dtype=np.uint8), -1)

        if frame is None:
            print("[ERROR] Cannot read image. Check path.")
            return

        result = process_frame(frame)
        cv2.imshow("YOLO Result", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode in ["video", "webcam"]:
        if args.mode == "video":
            cap = cv2.VideoCapture(args.path)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Cannot open video/webcam.")
            return

        print("[INFO] Starting... Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret: break

            result_frame = process_frame(frame)
            cv2.imshow("YOLO Live", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()