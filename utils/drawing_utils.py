# drawing_utils.py
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import PoseLandmarkerResult, HandLandmarkerResult # For type hints

def draw_pose_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draws pose landmarks and connections on an RGB image.

    Args:
        rgb_image (np.ndarray): The input image in RGB format.
        detection_result (PoseLandmarkerResult): The result from MediaPipe PoseLandmarker.

    Returns:
        np.ndarray: The image with pose landmarks drawn, in RGB format.
    """
    annotated_image = np.copy(rgb_image)
    if detection_result.pose_landmarks:
        for pose_landmarks_for_one_person in detection_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z,
                    visibility=landmark.visibility, presence=landmark.presence
                ) for landmark in pose_landmarks_for_one_person
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
    
    # Draw segmentation masks if available and enabled
    if hasattr(detection_result, 'segmentation_masks') and detection_result.segmentation_masks:
        for segmentation_mask in detection_result.segmentation_masks:
            mask_np = segmentation_mask.numpy_view()
            # Create a condition for the mask (e.g., confidence > 0.5)
            condition = np.stack((mask_np,) * 3, axis=-1) > 0.5
            visual_mask = np.zeros_like(annotated_image, dtype=np.uint8)
            visual_mask[condition] = [0, 0, 120]  # Dark Red for segmented area
            annotated_image = cv2.addWeighted(annotated_image, 1.0, visual_mask, 0.3, 0)
            
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draws hand landmarks, connections, and handedness on an RGB image.

    Args:
        rgb_image (np.ndarray): The input image in RGB format.
        detection_result (HandLandmarkerResult): The result from MediaPipe HandLandmarker.

    Returns:
        np.ndarray: The image with hand landmarks drawn, in RGB format.
    """
    annotated_image = np.copy(rgb_image)
    if detection_result.hand_landmarks:
        for i, hand_landmarks_for_one_hand in enumerate(detection_result.hand_landmarks):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks_for_one_hand
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
            # Draw handedness
            if detection_result.handedness and i < len(detection_result.handedness):
                handedness_categories = detection_result.handedness[i]
                if handedness_categories:
                    category = handedness_categories[0]
                    text = f"{category.category_name} ({category.score:.2f})"
                    if hand_landmarks_for_one_hand: # Ensure landmarks exist for positioning text
                        text_x = int(hand_landmarks_for_one_hand[0].x * annotated_image.shape[1])
                        text_y = int(hand_landmarks_for_one_hand[0].y * annotated_image.shape[0]) - 10
                        cv2.putText(annotated_image, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) # Black outline
                        cv2.putText(annotated_image, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA) # White text
    return annotated_image