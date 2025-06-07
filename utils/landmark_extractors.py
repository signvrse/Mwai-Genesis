# landmark_extractors.py
import cv2
import numpy as np
from typing import Tuple, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult

# --- Module-level Detector Instances and Model Paths ---
POSE_DETECTOR = None
HAND_DETECTOR = None

# Ensure these paths are correct relative to where you run main_processor.py,
# or use absolute paths.
MODEL_POSE_PATH = 'pose_landmarker_heavy.task'
MODEL_HAND_PATH = 'hand_landmarker.task'

def initialize_pose_detector(num_poses: int = 1, output_segmentation: bool = True):
    """
    Initializes the MediaPipe PoseLandmarker if it hasn't been already.

    Args:
        num_poses (int): Maximum number of poses to detect.
        output_segmentation (bool): Whether to output segmentation masks.

    Returns:
        Optional[vision.PoseLandmarker]: The initialized PoseLandmarker instance,
                                         or None if initialization fails.
    """
    global POSE_DETECTOR
    if POSE_DETECTOR is None:
        try:
            print("Initializing PoseLandmarker...")
            base_options = python.BaseOptions(model_asset_path=MODEL_POSE_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5, # More relevant for VIDEO mode
                output_segmentation_masks=output_segmentation
            )
            POSE_DETECTOR = vision.PoseLandmarker.create_from_options(options)
            print("PoseLandmarker initialized.")
        except Exception as e:
            print(f"Error initializing PoseLandmarker: {e}")
            POSE_DETECTOR = None
    return POSE_DETECTOR

def initialize_hand_detector(num_hands: int = 2):
    """
    Initializes the MediaPipe HandLandmarker if it hasn't been already.

    Args:
        num_hands (int): Maximum number of hands to detect.

    Returns:
        Optional[vision.HandLandmarker]: The initialized HandLandmarker instance,
                                         or None if initialization fails.
    """
    global HAND_DETECTOR
    if HAND_DETECTOR is None:
        try:
            print("Initializing HandLandmarker...")
            base_options = python.BaseOptions(model_asset_path=MODEL_HAND_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            HAND_DETECTOR = vision.HandLandmarker.create_from_options(options)
            print("HandLandmarker initialized.")
        except Exception as e:
            print(f"Error initializing HandLandmarker: {e}")
            HAND_DETECTOR = None
    return HAND_DETECTOR

def extract_pose_landmarks(cv_bgr_frame: np.ndarray):
    """
    Extracts pose landmarks from a single OpenCV BGR frame.
    Assumes initialize_pose_detector() has been called.

    Args:
        cv_bgr_frame (np.ndarray): The input image as a NumPy array (BGR format).

    Returns:
        Optional[PoseLandmarkerResult]: The detection result from PoseLandmarker,
                                        or None if detection fails or detector not initialized.
    """
    if POSE_DETECTOR is None:
        print("Pose detector not initialized. Call initialize_pose_detector() first.")
        return None

    rgb_frame = cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    try:
        detection_result = POSE_DETECTOR.detect(mp_image)
        return detection_result
    except Exception as e:
        print(f"Error during pose detection: {e}")
        return None

def extract_hand_landmarks(cv_bgr_frame: np.ndarray):
    """
    Extracts hand landmarks from a single OpenCV BGR frame.
    Assumes initialize_hand_detector() has been called.

    Args:
        cv_bgr_frame (np.ndarray): The input image as a NumPy array (BGR format).

    Returns:
        Optional[HandLandmarkerResult]: The detection result from HandLandmarker,
                                        or None if detection fails or detector not initialized.
    """
    if HAND_DETECTOR is None:
        print("Hand detector not initialized. Call initialize_hand_detector() first.")
        return None

    rgb_frame = cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    try:
        detection_result = HAND_DETECTOR.detect(mp_image)
        return detection_result
    except Exception as e:
        print(f"Error during hand detection: {e}")
        return None

def close_detectors():
    """Closes any initialized MediaPipe detectors."""
    global POSE_DETECTOR, HAND_DETECTOR
    if POSE_DETECTOR:
        print("Closing PoseLandmarker...")
        POSE_DETECTOR.close()
        POSE_DETECTOR = None
    if HAND_DETECTOR:
        print("Closing HandLandmarker...")
        HAND_DETECTOR.close()
        HAND_DETECTOR = None
    print("Detectors closed.")