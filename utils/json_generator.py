# # formatting_utils.py
# from typing import Dict, Any, List, Optional
# from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult # For type hints

# # --- Landmark Names (Constants) ---
# POSE_LANDMARK_NAMES = [
#     "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
#     "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
#     "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
#     "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
#     "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
#     "left_foot_index", "right_foot_index"
# ]

# HAND_LANDMARK_NAMES = [
#     "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
#     "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
#     "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
#     "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
#     "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
# ]

# def format_pose_results_to_dict(
#     detection_result
# ) -> List[Dict[str, Dict[str, float]]]:
#     """
#     Formats PoseLandmarkerResult into a list of pose dictionaries.

#     Each pose dictionary contains landmark names mapped to their normalized
#     x, y, z, visibility, and presence values.

#     Args:
#         detection_result (Optional[PoseLandmarkerResult]): The result from MediaPipe PoseLandmarker.
#                                                           Can be None if no detection occurred.

#     Returns:
#         List[Dict[str, Dict[str, float]]]: A list of dictionaries, where each
#                                            dictionary represents a detected pose.
#                                            Returns an empty list if no poses are detected or
#                                            if detection_result is None.
#     """
#     all_poses_data: List[Dict[str, Dict[str, float]]] = []
#     if not detection_result or not detection_result.pose_landmarks:
#         return all_poses_data

#     for person_landmarks in detection_result.pose_landmarks:
#         current_pose_dict: Dict[str, Dict[str, float]] = {}
#         for landmark_idx, landmark in enumerate(person_landmarks):
#             if landmark_idx < len(POSE_LANDMARK_NAMES):
#                 landmark_name = POSE_LANDMARK_NAMES[landmark_idx]
#                 current_pose_dict[landmark_name] = {
#                     "x": landmark.x,
#                     "y": landmark.y,
#                     "z": landmark.z,
#                     "visibility": landmark.visibility,
#                     "presence": landmark.presence
#                 }
#         if current_pose_dict: # Add only if landmarks were processed
#             all_poses_data.append(current_pose_dict)
#     return all_poses_data

# def format_hand_results_to_dict(
#     detection_result
# ) -> Dict[str, Optional[Dict[str, Dict[str, float]]]]:
#     """
#     Formats HandLandmarkerResult into a dictionary with "left" and "right" hand data.

#     Each hand's data is a dictionary of landmark names mapped to their normalized
#     x, y, z values (and placeholder visibility/presence as per original request).

#     Args:
#         detection_result (Optional[HandLandmarkerResult]): The result from MediaPipe HandLandmarker.
#                                                            Can be None if no detection occurred.

#     Returns:
#         Dict[str, Optional[Dict[str, Dict[str, float]]]]:
#             A dictionary with "left" and "right" keys. Each key holds either
#             None (if the hand is not detected/result is None) or a dictionary of
#             landmark names mapped to their data.
#     """
#     output_data: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {
#         "left": None,
#         "right": None
#     }
#     if not detection_result or not detection_result.hand_landmarks:
#         return output_data

#     for i, handedness_categories in enumerate(detection_result.handedness):
#         if not handedness_categories:
#             continue
#         hand_label = handedness_categories[0].category_name.lower() # "left" or "right"

#         if hand_label not in output_data:
#             print(f"Warning: Unexpected hand label '{hand_label}' found in formatting.")
#             continue
        
#         current_hand_normalized_landmarks = detection_result.hand_landmarks[i]
#         hand_landmarks_dict: Dict[str, Dict[str, float]] = {}

#         for landmark_idx, landmark in enumerate(current_hand_normalized_landmarks):
#             if landmark_idx < len(HAND_LANDMARK_NAMES):
#                 landmark_name = HAND_LANDMARK_NAMES[landmark_idx]
#                 hand_landmarks_dict[landmark_name] = {
#                     "x": landmark.x,
#                     "y": landmark.y,
#                     "z": landmark.z,
#                     # As discussed, visibility/presence are not standard attributes for
#                     # HandLandmarker's NormalizedLandmark objects within `hand_landmarks`.
#                     # Including them as 0.0 based on the original implied structure.
#                     "visibility": getattr(landmark, 'visibility', 0.0),
#                     "presence": getattr(landmark, 'presence', 0.0),
#                 }
#         if hand_landmarks_dict: # Add only if landmarks were processed
#             output_data[hand_label] = hand_landmarks_dict
            
#     return output_data

# formatting_utils.py
# formatting_utils.py
from typing import Dict, Any, List, Optional
import numpy as np # For vector operations if needed, though direct component-wise is fine
from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult # For type hints

# --- Landmark Names (Constants) ---
# Original Pose Landmark Names (Indices match MediaPipe output)
ORIGINAL_POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

HAND_LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

def _calculate_midpoint(p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
    """Helper function to calculate the midpoint between two landmark points."""
    return {
        "x": (p1["x"] + p2["x"]) / 2,
        "y": (p1["y"] + p2["y"]) / 2,
        "z": (p1["z"] + p2["z"]) / 2,
        "visibility": (p1["visibility"] + p2["visibility"]) / 2,
        "presence": (p1["presence"] + p2["presence"]) / 2,
    }

def _interpolate_point(p1: Dict[str, float], p2: Dict[str, float], fraction: float) -> Dict[str, float]:
    """Helper function to interpolate a point between p1 and p2 at a given fraction from p1."""
    return {
        "x": p1["x"] + fraction * (p2["x"] - p1["x"]),
        "y": p1["y"] + fraction * (p2["y"] - p1["y"]),
        "z": p1["z"] + fraction * (p2["z"] - p1["z"]),
        "visibility": p1["visibility"] + fraction * (p2["visibility"] - p1["visibility"]),
        "presence": p1["presence"] + fraction * (p2["presence"] - p1["presence"]),
    }


def format_pose_results_to_dict(
    detection_result: Optional[PoseLandmarkerResult]
) -> List[Dict[str, Dict[str, float]]]:
    """
    Formats PoseLandmarkerResult into a list of pose dictionaries,
    including derived points like 'hips', 'neck', 'spine' points,
    'left_arm', 'right_arm', 'left_forearm', 'right_forearm'.

    - "left_hip" and "right_hip" are replaced by a single "hips" midpoint.
    - "neck" is added as the midpoint of "left_shoulder" and "right_shoulder".
    - "spine", "spine1", "spine2" are added, equally dividing the segment
      between the new "hips" and "neck" points.
    - "left_arm" is the midpoint of "left_shoulder" and "left_elbow".
    - "right_arm" is the midpoint of "right_shoulder" and "right_elbow".
    - "left_forearm" is the midpoint of "left_elbow" and "left_wrist".
    - "right_forearm" is the midpoint of "right_elbow" and "right_wrist".
    - "left_elbow" and "right_elbow" are REMOVED from the final output.

    Args:
        detection_result (Optional[PoseLandmarkerResult]): The result from MediaPipe PoseLandmarker.

    Returns:
        List[Dict[str, Dict[str, float]]]: A list of dictionaries, where each
                                           dictionary represents a detected pose
                                           with original and derived landmarks.
                                           Returns an empty list if no poses are
                                           detected or if detection_result is None.
    """
    all_formatted_poses_data: List[Dict[str, Dict[str, float]]] = []
    if not detection_result or not detection_result.pose_landmarks:
        return all_formatted_poses_data

    for person_landmarks in detection_result.pose_landmarks:
        # 1. Convert original landmarks to a temporary dictionary for easy access
        temp_original_landmarks_dict: Dict[str, Dict[str, float]] = {}
        for landmark_idx, landmark in enumerate(person_landmarks):
            if landmark_idx < len(ORIGINAL_POSE_LANDMARK_NAMES):
                landmark_name = ORIGINAL_POSE_LANDMARK_NAMES[landmark_idx]
                temp_original_landmarks_dict[landmark_name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "presence": landmark.presence
                }
        
        if not temp_original_landmarks_dict:
            continue

        # Initialize the output dictionary for the current person
        # Start by copying all original landmarks that are not being replaced or used for derivation
        # We will explicitly remove elbows later.
        current_person_output_dict: Dict[str, Dict[str, float]] = {
            name: data for name, data in temp_original_landmarks_dict.items()
            if name not in ["left_hip", "right_hip"] # These will be replaced by "hips"
        }

        # 2. Calculate "hips" midpoint and replace left/right hip
        left_hip_data = temp_original_landmarks_dict.get("left_hip")
        right_hip_data = temp_original_landmarks_dict.get("right_hip")
        hips_midpoint_data = None
        if left_hip_data and right_hip_data:
            hips_midpoint_data = _calculate_midpoint(left_hip_data, right_hip_data)
            current_person_output_dict["hips"] = hips_midpoint_data
        else:
            print("Warning: Missing left or right hip data for a person. 'hips' and spine points might be inaccurate or missing.")

        # 3. Calculate "neck" midpoint (from shoulders) and add it
        left_shoulder_data = temp_original_landmarks_dict.get("left_shoulder")
        right_shoulder_data = temp_original_landmarks_dict.get("right_shoulder")
        neck_midpoint_data = None
        if left_shoulder_data and right_shoulder_data:
            neck_midpoint_data = _calculate_midpoint(left_shoulder_data, right_shoulder_data)
            current_person_output_dict["neck"] = neck_midpoint_data
        else:
            print("Warning: Missing left or right shoulder data for a person. 'neck' and spine points might be inaccurate or missing.")

        # 4. Calculate spine points between new "hips" and "neck"
        if hips_midpoint_data and neck_midpoint_data:
            current_person_output_dict["spine"] = _interpolate_point(hips_midpoint_data, neck_midpoint_data, 0.25)
            current_person_output_dict["spine1"] = _interpolate_point(hips_midpoint_data, neck_midpoint_data, 0.50)
            current_person_output_dict["spine2"] = _interpolate_point(hips_midpoint_data, neck_midpoint_data, 0.75)
        else:
            print("Warning: Could not calculate spine points due to missing hips or neck data.")

        # 5. Calculate arm and forearm midpoints
        # Left Arm/Forearm
        left_elbow_data = temp_original_landmarks_dict.get("left_elbow")
        left_wrist_data = temp_original_landmarks_dict.get("left_wrist")
        if left_shoulder_data and left_elbow_data:
            current_person_output_dict["left_arm"] = _calculate_midpoint(left_shoulder_data, left_elbow_data)
        else:
            print("Warning: Missing left_shoulder or left_elbow for left_arm calculation.")
        if left_elbow_data and left_wrist_data:
            current_person_output_dict["left_forearm"] = _calculate_midpoint(left_elbow_data, left_wrist_data)
        else:
            print("Warning: Missing left_elbow or left_wrist for left_forearm calculation.")

        # Right Arm/Forearm
        right_elbow_data = temp_original_landmarks_dict.get("right_elbow")
        right_wrist_data = temp_original_landmarks_dict.get("right_wrist")
        if right_shoulder_data and right_elbow_data:
            current_person_output_dict["right_arm"] = _calculate_midpoint(right_shoulder_data, right_elbow_data)
        else:
            print("Warning: Missing right_shoulder or right_elbow for right_arm calculation.")
        if right_elbow_data and right_wrist_data:
            current_person_output_dict["right_forearm"] = _calculate_midpoint(right_elbow_data, right_wrist_data)
        else:
            print("Warning: Missing right_elbow or right_wrist for right_forearm calculation.")
            
        # 6. Remove original elbow points
        current_person_output_dict.pop("left_elbow", None) # Use .pop with default None in case it was already missing
        current_person_output_dict.pop("right_elbow", None)


        if current_person_output_dict:
            all_formatted_poses_data.append(current_person_output_dict)
            
    return all_formatted_poses_data


def format_hand_results_to_dict(
    detection_result: Optional[HandLandmarkerResult]
) -> Dict[str, Optional[Dict[str, Dict[str, float]]]]:
    """
    Formats HandLandmarkerResult into a dictionary with "left" and "right" hand data.

    Each hand's data is a dictionary of landmark names mapped to their normalized
    x, y, z values (and placeholder visibility/presence as per original request).

    Args:
        detection_result (Optional[HandLandmarkerResult]): The result from MediaPipe HandLandmarker.
                                                           Can be None if no detection occurred.

    Returns:
        Dict[str, Optional[Dict[str, Dict[str, float]]]]:
            A dictionary with "left" and "right" keys. Each key holds either
            None (if the hand is not detected/result is None) or a dictionary of
            landmark names mapped to their data.
    """
    output_data: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {
        "left": None,
        "right": None
    }
    if not detection_result or not detection_result.hand_landmarks:
        return output_data

    for i, handedness_categories in enumerate(detection_result.handedness):
        if not handedness_categories:
            continue
        hand_label = handedness_categories[0].category_name.lower() # "left" or "right"

        if hand_label not in output_data:
            print(f"Warning: Unexpected hand label '{hand_label}' found in formatting.")
            continue
        
        current_hand_normalized_landmarks = detection_result.hand_landmarks[i]
        hand_landmarks_dict: Dict[str, Dict[str, float]] = {}

        for landmark_idx, landmark in enumerate(current_hand_normalized_landmarks):
            if landmark_idx < len(HAND_LANDMARK_NAMES):
                landmark_name = HAND_LANDMARK_NAMES[landmark_idx]
                hand_landmarks_dict[landmark_name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": getattr(landmark, 'visibility', 0.0),
                    "presence": getattr(landmark, 'presence', 0.0),
                }
        if hand_landmarks_dict: # Add only if landmarks were processed
            output_data[hand_label] = hand_landmarks_dict
            
    return output_data