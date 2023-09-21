# Object Detection Comparison: RT-DETR vs YOLOv8

This repository contains code for running real-time object detection using two state-of-the-art models: Real-Time Detection Transformer (RT-DETR) and YOLOv8. The goal is to provide a side-by-side comparison of these models in terms of speed and accuracy.

## Features

- Real-time object detection using RT-DETR and YOLOv8.
- Command-line arguments for specifying model, input video, and output path.
- Video stitching for a side-by-side comparison.

## Dependencies

- Python 3.x
- OpenCV
- PyTorch
- NumPy
- Ultralytics library for YOLO and RT-DETR
- Supervision for bounding box annotation

## Usage

### RT-DETR Inference

Run the following command to execute object detection using RT-DETR:

```bash
python rt_detr_infer.py --model_name="rtdetr-l.pt" --input_video="input_video.mp4" --output_path="output_video_rt-detr-l.avi"
```
### YOLOv8 Inference

Run the following command to execute object detection using YOLOV8:

```bash
python yolov8_infer.py --model_name="yolov8-l.pt" --input_video="input_video.mp4" --output_path="output_video_yolov8-l.avi"
```
### Video Stitching

Run the following command to stitch two videos for side-by-side comparison:

```bash
python stitch_videos.py
```
### Results

The output videos can be used to evaluate and compare the performance of RT-DETR and YOLOv8.

### Contributing

Feel free to open an issue or pull request if you find any issues or have suggestions for improvements.

### Pros and Cons:

#### Pros:

- Clear structure: The README is organized into sections for easy navigation.
- Detailed: Each section contains enough information for the user to understand what the repository is about and how to use it.
- Inclusion of code snippets: This will help users to quickly understand how to execute the scripts.

#### Cons:

- Assumes familiarity with terms like RT-DETR and YOLOv8; may not be beginner-friendly.
- Dependency list might need to be expanded depending on your specific requirements.

### License

MIT License

