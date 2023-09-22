from typing import Any
import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR
import supervision as sv
import argparse


# Initialize argument parser
parser = argparse.ArgumentParser(description="Object Detection using RT-DETR")
parser.add_argument("--model_name", default="rtdetr-l.pt", type=str, help="Model name to load for RT-DETR")
parser.add_argument("--input_video", required=True, type=str, help="Path to input video file")
parser.add_argument("--output_path", default="output_video_rt-detr-l.avi", type=str, help="Path to save the output video")
args = parser.parse_args()


COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class DETRClass:
    def __init__(self, video_path, model_name, output_file_path):
        self.video_path = video_path
        self.output_file_path = output_file_path

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using device: ", self.device)

        self.model = RTDETR(model_name)
        self.CLASS_NAMES_DICT = COCO_CLASSES

        print("Classes: ", self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.5)

    def plot_bboxes(self, results, frame):
        # Extract the results
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
        # Filter based on confidence threshold
        valid_indices = np.where(conf >= 0.40)[0]
        class_id = class_id[valid_indices]
        xyxy = xyxy[valid_indices]
        conf = conf[valid_indices]

        # Plot the bounding boxes
        class_id = class_id.astype(np.int32)
        detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=conf)
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}" for xyxy, mask, confidence, class_id, track_id in detections]
        frame = self.box_annotator.annotate(frame, detections, self.labels)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.video_path)

        assert cap.isOpened()
        
        # Get video properties: width, height, and frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize the VideoWriter 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time.perf_counter()
            frame_fps = 1 / (end_time - start_time)

            # Draw filled rectangle for FPS
            fps_text = f"FPS: {frame_fps:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            top_left = (10, 70)
            bottom_right = (10 + text_width + 10, 70 + text_height + 20)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)
            cv2.putText(frame, fps_text, (20, 70 + (text_height + 20) // 2 + text_height // 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Draw filled rectangle for "RT_DETR-l"
            model_text = "RT_DETR-l"
            (text_width, text_height), _ = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            top_left = (10, 95 + text_height)
            bottom_right = (10 + text_width + 10, 95 + 2*text_height + 10)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)
            cv2.putText(frame, model_text, (20, 95 + text_height + (text_height + 10) // 2 + text_height // 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()

        

# Modify this line to use CLI argument
video_file_path = args.input_video  # This will now come from the CLI argument
output_file_path = args.output_path  # This will now come from the CLI argument
model_name = args.model_name  # This will now come from the CLI argument

transformer_detector = DETRClass(video_file_path, model_name, output_file_path)
transformer_detector()
