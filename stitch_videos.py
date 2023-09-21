import cv2
import numpy as np

def stitch_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Desired output size
    OUTPUT_HEIGHT = 1040
    SEPARATOR_WIDTH = 10  # Width of the black separator in pixels

    # Calculate the scaling factors for each video while preserving their aspect ratios
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale1 = OUTPUT_HEIGHT / height1
    scale2 = OUTPUT_HEIGHT / height2

    new_width1 = int(width1 * scale1)
    new_width2 = int(width2 * scale2)

    # Initialize VideoWriter for the stitched video
    OUTPUT_WIDTH = new_width1 + SEPARATOR_WIDTH + new_width2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, int(cap1.get(cv2.CAP_PROP_FPS)), (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize the frames
        frame1 = cv2.resize(frame1, (new_width1, OUTPUT_HEIGHT))
        frame2 = cv2.resize(frame2, (new_width2, OUTPUT_HEIGHT))

        # Create a black separator
        separator = np.zeros((OUTPUT_HEIGHT, SEPARATOR_WIDTH, 3), dtype=np.uint8)

        # Concatenate the frames with the separator in between
        combined_frame = np.hstack((frame1, separator, frame2))
        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()

# Use the function
stitch_videos('output_video_rt-detr-l.avi', 'output_video_yolov8-l.avi', 'comparison_video.avi')
