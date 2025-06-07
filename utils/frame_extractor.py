# video_utils.py
import cv2
import numpy as np
from typing import List

def video_to_frames(video_path: str) -> List[np.ndarray]:
    """
    Reads a video file and extracts all its frames into a list of NumPy arrays.

    Each frame is in BGR color format by default from OpenCV.

    Args:
        video_path (str): The path to the video file.

    Returns:
        List[np.ndarray]: A list of frames. Returns an empty list if the video
                          cannot be opened or no frames are read.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames

    print(f"Processing video: {video_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or error
        frames.append(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Read {frame_count} frames...")

    cap.release()

    if frame_count > 0:
        print(f"Successfully extracted {len(frames)} frames from {video_path}.")
    else:
        print(f"No frames were read from {video_path}. The video might be empty or corrupted.")
    return frames