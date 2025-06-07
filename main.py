# import json
# import cv2
# import numpy as np
# from typing import List, Tuple, Optional, Dict # Added Tuple and Optional

# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# # import numpy as np # Already imported

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # --- Global Detector Instances (Initialize ONCE) ---
# POSE_DETECTOR: Optional[vision.PoseLandmarker] = None
# HAND_DETECTOR: Optional[vision.HandLandmarker] = None

# MODEL_POSE_PATH = 'pose_landmarker_heavy.task' # Make sure this exists
# MODEL_HAND_PATH = 'hand_landmarker.task'     # Make sure this exists

# # --- Video to Frames Function (Unchanged, looks good) ---
# def video_to_frames(video_path: str) -> List[np.ndarray]:
#     frames: List[np.ndarray] = []
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return frames
#     print(f"Processing video: {video_path}")
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#         frame_count += 1
#         if frame_count % 100 == 0:
#             print(f"Read {frame_count} frames...")
#     cap.release()
#     if frame_count > 0:
#         print(f"Successfully extracted {len(frames)} frames from {video_path}.")
#     else:
#         print(f"No frames were read from {video_path}. The video might be empty or corrupted.")
#     return frames

# # --- Drawing Function for POSE Landmarks ---
# def draw_pose_landmarks_on_image(rgb_image: np.ndarray, detection_result: vision.PoseLandmarkerResult) -> np.ndarray:
#     annotated_image = np.copy(rgb_image)
#     if detection_result.pose_landmarks:
#         for pose_landmarks in detection_result.pose_landmarks:
#             pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             pose_landmarks_proto.landmark.extend([
#                 landmark_pb2.NormalizedLandmark(
#                     x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility
#                 ) for landmark in pose_landmarks
#             ])
#             solutions.drawing_utils.draw_landmarks(
#                 annotated_image,
#                 pose_landmarks_proto,
#                 solutions.pose.POSE_CONNECTIONS,
#                 solutions.drawing_styles.get_default_pose_landmarks_style()
#             )
#     # You could also draw segmentation masks here if needed using detection_result.segmentation_masks
#     return annotated_image

# # --- Drawing Function for HAND Landmarks ---
# def draw_hand_landmarks_on_image(rgb_image: np.ndarray, detection_result: vision.HandLandmarkerResult) -> np.ndarray:
#     annotated_image = np.copy(rgb_image)
#     if detection_result.hand_landmarks:
#         for i, hand_landmarks_for_one_hand in enumerate(detection_result.hand_landmarks):
#             hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             hand_landmarks_proto.landmark.extend([
#                 landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_for_one_hand
#             ])
#             solutions.drawing_utils.draw_landmarks(
#                 annotated_image,
#                 hand_landmarks_proto,
#                 solutions.hands.HAND_CONNECTIONS,
#                 solutions.drawing_styles.get_default_hand_landmarks_style(),
#                 solutions.drawing_styles.get_default_hand_connections_style()
#             )
#             # Draw handedness
#             if detection_result.handedness and i < len(detection_result.handedness):
#                 handedness_categories = detection_result.handedness[i]
#                 if handedness_categories:
#                     category = handedness_categories[0] # Assuming the first one is the most confident
#                     text = f"{category.category_name} ({category.score:.2f})"
#                     # Position text near the wrist (landmark 0)
#                     if hand_landmarks_for_one_hand:
#                         text_x = int(hand_landmarks_for_one_hand[0].x * annotated_image.shape[1])
#                         text_y = int(hand_landmarks_for_one_hand[0].y * annotated_image.shape[0]) - 10
#                         cv2.putText(annotated_image, text, (text_x, text_y),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
#                         cv2.putText(annotated_image, text, (text_x, text_y),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
#     return annotated_image

# # --- Initialize POSE Detector ---
# def initialize_pose_detector():
#     global POSE_DETECTOR
#     if POSE_DETECTOR is None:
#         print("Initializing PoseLandmarker...")
#         base_options = python.BaseOptions(model_asset_path=MODEL_POSE_PATH)
#         options = vision.PoseLandmarkerOptions(
#             base_options=base_options,
#             running_mode=vision.RunningMode.IMAGE, # Explicitly set
#             output_segmentation_masks=True, # Keep if you need masks
#             num_poses=1 # Adjust if you expect more poses
#         )
#         POSE_DETECTOR = vision.PoseLandmarker.create_from_options(options)
#         print("PoseLandmarker initialized.")
#     return POSE_DETECTOR

# # --- Initialize HAND Detector ---
# def initialize_hand_detector():
#     global HAND_DETECTOR
#     if HAND_DETECTOR is None:
#         print("Initializing HandLandmarker...")
#         base_options = python.BaseOptions(model_asset_path=MODEL_HAND_PATH)
#         options = vision.HandLandmarkerOptions(
#             base_options=base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,  # Detect up to 2 hands
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5 # Though less relevant for IMAGE mode
#         )
#         HAND_DETECTOR = vision.HandLandmarker.create_from_options(options)
#         print("HandLandmarker initialized.")
#     return HAND_DETECTOR

# # --- POSE Landmark Extractor ---
# def pose_landmark_extractor(cv_bgr_frame: np.ndarray) -> np.ndarray:
#     detector = initialize_pose_detector()
#     if not detector:
#         print("Pose detector not available.")
#         return cv_bgr_frame # Return original frame if detector failed to init

#     rgb_frame = cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

#     annotated_image_bgr = cv_bgr_frame # Default
#     try:
#         detection_result = detector.detect(mp_image)
#         if detection_result and detection_result.pose_landmarks:
#             print(f"  Pose landmarks found: {len(detection_result.pose_landmarks)} pose(s)")
#             annotated_rgb_image = draw_pose_landmarks_on_image(rgb_frame, detection_result)
#             annotated_image_bgr = cv2.cvtColor(annotated_rgb_image, cv2.COLOR_RGB2BGR)
#         else:
#             print("  No pose landmarks detected.")
#             # annotated_image_bgr remains the original cv_bgr_frame
#     except Exception as e:
#         print(f"Error in pose_landmark_extractor: {e}")
#         # annotated_image_bgr remains the original cv_bgr_frame

#     return detection_result

# # --- HAND Landmark Extractor ---
# def hand_landmark_extractor(cv_bgr_frame: np.ndarray):
#     detector = initialize_hand_detector()
#     if not detector:
#         print("Hand detector not available.")
#         return cv_bgr_frame

#     rgb_frame = cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
#     print(f"Processing frame for hand landmarks...") # Removed "got here" for clearer log
#     annotated_image_bgr = cv_bgr_frame # Default
#     try:
#         detection_result = detector.detect(mp_image)
#         if detection_result and detection_result.hand_landmarks:
#             print(f"  Hand landmarks found: {len(detection_result.hand_landmarks)} hand(s)")
#             annotated_rgb_image = draw_hand_landmarks_on_image(rgb_frame, detection_result)
#             annotated_image_bgr = cv2.cvtColor(annotated_rgb_image, cv2.COLOR_RGB2BGR)
#         else:
#             print("  No hand landmarks detected.")
#             # annotated_image_bgr remains the original cv_bgr_frame
#     except Exception as e:
#         print(f"Error in hand_landmark_extractor: {e}")
#         # annotated_image_bgr remains the original cv_bgr_frame
        
#     return detection_result

# LANDMARK_NAMES = [
#     "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
#     "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
#     "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
#     "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
#     "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
# ]

# def format_hand_landmarks_to_json_output(
#     detection_result
# ):
#     output_data: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {
#         "left": None,
#         "right": None
#     }
#     if detection_result.hand_landmarks:
#         for i, handedness_categories in enumerate(detection_result.handedness):
#             if not handedness_categories: continue
#             hand_label = handedness_categories[0].category_name.lower()
#             if hand_label not in output_data: continue
            
#             current_hand_landmarks = detection_result.hand_landmarks[i]
#             hand_landmarks_dict: Dict[str, Dict[str, float]] = {}
#             for landmark_idx, landmark in enumerate(current_hand_landmarks):
#                 if landmark_idx < len(LANDMARK_NAMES):
#                     landmark_name = LANDMARK_NAMES[landmark_idx]
#                     hand_landmarks_dict[landmark_name] = {
#                         "x": landmark.x,
#                         "y": landmark.y,
#                         "z": landmark.z,
#                         # As discussed, visibility/presence are not standard for HandLandmarker's NormalizedLandmark
#                         # but included based on previous output format of user.
#                         "visibility": landmark.visibility if hasattr(landmark, 'visibility') else 0.0,
#                         "presence": landmark.presence if hasattr(landmark, 'presence') else 0.0,
#                     }
#             output_data[hand_label] = hand_landmarks_dict
#     return output_data





# if __name__ == "__main__":
#     video_path = "./videos/video1.mp4"
    
#     # Ensure your .task files are in the same directory or update paths in
#     # MODEL_POSE_PATH and MODEL_HAND_PATH
#     all_frames_landmark_data = [] # List to store landmark data for all frames
#     extracted_frames = video_to_frames(video_path)
#     output_json_file_path = "video_hand_landmarks.json" # Path to save the JSON output

#     if extracted_frames:
#         # Choose which extractor to use:
#         USE_POSE_EXTRACTOR = True  # Set to True for pose, False for hand
        
#         # window_name = "Annotated Video Frame"
#         # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Create window once

#         for idx, frame in enumerate(extracted_frames):
#             print(f"\nProcessing Frame {idx + 1}/{len(extracted_frames)}")
            
#             if USE_POSE_EXTRACTOR:
#                 # pose_landmarks = pose_landmark_extractor(frame)
#                 pose_landmarks = hand_landmark_extractor(frame)

#                 frame_landmark_output = {
#                 "frame_number": idx,
#                 "hands": {"left": None, "right": None} # Default if no detection_result
#                 }

#                 if pose_landmarks:
#                     # Format the current frame's hand landmarks
#                     formatted_hands_data = format_hand_landmarks_to_json_output(pose_landmarks)
#                     frame_landmark_output["hands"] = formatted_hands_data
                
                
#                 all_frames_landmark_data.append(frame_landmark_output)


#             else:
#                 annotated_image = hand_landmark_extractor(frame)
                
#                 print("-"*20)
#                 print(annotated_image)
#                 print("-"*20)

#         print(all_frames_landmark_data)

#         try:
#             with open(output_json_file_path, 'w') as f_json:
#                 json.dump(all_frames_landmark_data, f_json, indent=2)
#             print(f"\nSuccessfully saved hand landmark data for all frames to: {output_json_file_path}")
#         except IOError as e:
#             print(f"Error writing JSON file: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred while saving JSON: {e}")
#             # cv2.imshow(window_name, annotated_image)
            
#             # Wait for 30ms, or until 'q' is pressed
#             # Adjust delay for desired playback speed (1000ms / desired_fps)
#         #     key = cv2.waitKey(1000) & 0xFF
#         #     if key == ord('q'):
#         #         print("Quitting...")
#         #         break
        
#         # cv2.destroyAllWindows()

#         # --- Close detectors ---
#         if POSE_DETECTOR:
#             print("Closing PoseLandmarker...")
#             POSE_DETECTOR.close()
#         if HAND_DETECTOR:
#             print("Closing HandLandmarker...")
#             HAND_DETECTOR.close()
#         print("Done.")

#     else:
#         print(f"No frames extracted from {video_path}. Exiting.")

# main_processor.py
import cv2
import numpy as np
import json
from typing import List, Dict, Any

# Import functions from our custom modules
from utils.frame_extractor import video_to_frames
from utils.landmark_extractors import (
    initialize_pose_detector,
    initialize_hand_detector,
    extract_pose_landmarks,
    extract_hand_landmarks,
    close_detectors
)
from utils.drawing_utils import (
    draw_pose_landmarks_on_image,
    draw_hand_landmarks_on_image
)
from utils.json_generator import (
    format_pose_results_to_dict,
    format_hand_results_to_dict
)

def main():
    """
    Main function to process a video, extract pose and hand landmarks,
    and save the combined results to a JSON file.
    """
    video_path = "./videos/video1.mp4"  # Ensure this video exists
    output_json_path = "video_all_landmarks_output_2.json"
    
    # --- Configuration ---
    run_pose_detection = True
    run_hand_detection = True
    display_annotated_frames = False # Set to True to see live annotations
    num_poses_to_detect = 2      # Max poses for pose detector
    num_hands_to_detect = 2      # Max hands for hand detector
    output_segmentation_masks_for_pose = True # For pose detector
    # ---------------------

    all_frames_data: List[Dict[str, Any]] = []
    extracted_cv_frames = video_to_frames(video_path)

    if not extracted_cv_frames:
        print(f"No frames extracted from {video_path}. Exiting.")
        return

    # Initialize detectors once
    if run_pose_detection:
        initialize_pose_detector(num_poses=num_poses_to_detect, output_segmentation=output_segmentation_masks_for_pose)
    if run_hand_detection:
        initialize_hand_detector(num_hands=num_hands_to_detect)

    window_name = "Annotated Frame"
    if display_annotated_frames:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for frame_idx, bgr_frame in enumerate(extracted_cv_frames):
        print(f"\nProcessing Frame {frame_idx + 1}/{len(extracted_cv_frames)}")
        
        current_frame_data: Dict[str, Any] = {
            "frame_number": frame_idx,
            "poses": [],       # Default to empty list
            "hands": {"left": None, "right": None} # Default
        }
        
        # Create a copy of the frame for annotation to avoid modifying original
        annotated_bgr_frame = bgr_frame.copy() 

        # --- Pose Detection and Formatting ---
        if run_pose_detection:
            pose_detection_result = extract_pose_landmarks(bgr_frame)
            if pose_detection_result:
                current_frame_data["poses"] = format_pose_results_to_dict(pose_detection_result)
                if pose_detection_result.pose_landmarks:
                    print(f"  Detected {len(pose_detection_result.pose_landmarks)} pose(s).")
                    if display_annotated_frames: # Annotate only if displaying
                        rgb_frame_for_pose_drawing = cv2.cvtColor(annotated_bgr_frame, cv2.COLOR_BGR2RGB)
                        annotated_rgb_frame = draw_pose_landmarks_on_image(rgb_frame_for_pose_drawing, pose_detection_result)
                        annotated_bgr_frame = cv2.cvtColor(annotated_rgb_frame, cv2.COLOR_RGB2BGR)
                else:
                    print("  No poses detected in this frame.")
            else:
                print("  Pose detection failed or no result for this frame.")

        # --- Hand Detection and Formatting ---
        if run_hand_detection:
            hand_detection_result = extract_hand_landmarks(bgr_frame)
            if hand_detection_result:
                current_frame_data["hands"] = format_hand_results_to_dict(hand_detection_result)
                if hand_detection_result.hand_landmarks:
                    print(f"  Detected {len(hand_detection_result.hand_landmarks)} hand(s).")
                    if display_annotated_frames: # Annotate only if displaying
                        rgb_frame_for_hand_drawing = cv2.cvtColor(annotated_bgr_frame, cv2.COLOR_BGR2RGB)
                        annotated_rgb_frame = draw_hand_landmarks_on_image(rgb_frame_for_hand_drawing, hand_detection_result)
                        annotated_bgr_frame = cv2.cvtColor(annotated_rgb_frame, cv2.COLOR_RGB2BGR)
                else:
                    print("  No hands detected in this frame.")
            else:
                print("  Hand detection failed or no result for this frame.")
        
        all_frames_data.append(current_frame_data)

        if display_annotated_frames:
            cv2.imshow(window_name, annotated_bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Very short delay, press 'q' to quit
                print("User quit display.")
                break
    
    if display_annotated_frames:
        cv2.destroyAllWindows()

    # --- Save combined data to JSON ---
    try:
        with open(output_json_path, 'w') as f_json:
            json.dump(all_frames_data, f_json, indent=2)
        print(f"\nSuccessfully saved all landmark data to: {output_json_path}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving JSON: {e}")
    finally:
        # --- Clean up detectors ---
        close_detectors()
        print("Processing complete.")

if __name__ == "__main__":
    main()