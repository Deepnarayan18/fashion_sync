import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_distance  # Changed from 'utils' to '.utils'

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmark_names = self.mp_pose.PoseLandmark

    def process_frame(self, frame, is_single_image=False):
        """Process a frame or image and return pose landmarks."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        if is_single_image:
            return results.pose_landmarks, image
        return results.pose_landmarks

    def get_measurements(self, landmarks, frame_shape):
        """Extract detailed body measurements."""
        if not landmarks:
            return None
        
        def to_pixels(landmark):
            return [landmark.x * frame_shape[1], landmark.y * frame_shape[0]]

        # Keypoints
        left_shoulder = to_pixels(landmarks.landmark[self.landmark_names.LEFT_SHOULDER])
        right_shoulder = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_SHOULDER])
        left_elbow = to_pixels(landmarks.landmark[self.landmark_names.LEFT_ELBOW])
        right_elbow = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_ELBOW])
        left_wrist = to_pixels(landmarks.landmark[self.landmark_names.LEFT_WRIST])
        right_wrist = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_WRIST])
        left_hip = to_pixels(landmarks.landmark[self.landmark_names.LEFT_HIP])
        right_hip = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_HIP])
        left_knee = to_pixels(landmarks.landmark[self.landmark_names.LEFT_KNEE])
        right_knee = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_KNEE])
        left_ankle = to_pixels(landmarks.landmark[self.landmark_names.LEFT_ANKLE])
        right_ankle = to_pixels(landmarks.landmark[self.landmark_names.RIGHT_ANKLE])

        # Measurements
        measurements = {
            "shoulder_width": calculate_distance(left_shoulder, right_shoulder),
            "torso_length": (calculate_distance(left_shoulder, left_hip) + 
                             calculate_distance(right_shoulder, right_hip)) / 2,
            "hip_width": calculate_distance(left_hip, right_hip),
            "waist_width": (calculate_distance(left_hip, right_hip) * 0.8),
            "left_arm_length": (calculate_distance(left_shoulder, left_elbow) + 
                                calculate_distance(left_elbow, left_wrist)),
            "right_arm_length": (calculate_distance(right_shoulder, right_elbow) + 
                                 calculate_distance(right_elbow, right_wrist)),
            "left_leg_length": (calculate_distance(left_hip, left_knee) + 
                                calculate_distance(left_knee, left_ankle)),
            "right_leg_length": (calculate_distance(right_hip, right_knee) + 
                                 calculate_distance(right_knee, right_ankle)),
            "shoulder_to_hip_ratio": calculate_distance(left_shoulder, right_shoulder) / 
                                     calculate_distance(left_hip, right_hip),
            "torso_to_leg_ratio": ((calculate_distance(left_shoulder, left_hip) + 
                                    calculate_distance(right_shoulder, right_hip)) / 2) / 
                                   ((calculate_distance(left_hip, left_ankle) + 
                                     calculate_distance(right_hip, right_ankle)) / 2)
        }
        return measurements

    def determine_body_type(self, measurements):
        """Classify body type with detailed criteria."""
        shoulder_to_hip = measurements["shoulder_to_hip_ratio"]
        torso_to_leg = measurements["torso_to_leg_ratio"]
        hip_width = measurements["hip_width"]
        shoulder_width = measurements["shoulder_width"]

        if shoulder_to_hip > 1.1 and torso_to_leg > 0.8:
            return "Inverted Triangle"
        elif shoulder_to_hip < 0.9 and hip_width > shoulder_width * 1.05:
            return "Pear"
        elif abs(shoulder_to_hip - 1.0) < 0.1 and abs(hip_width - shoulder_width) < 20:
            return "Rectangle"
        elif shoulder_to_hip >= 0.9 and shoulder_to_hip <= 1.1 and hip_width > shoulder_width * 1.1:
            return "Hourglass"
        elif torso_to_leg < 0.7 and hip_width > shoulder_width:
            return "Apple"
        else:
            return "Oval"

    def cleanup(self):
        """Release resources."""
        self.pose.close()