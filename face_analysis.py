import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_distance

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, 
                                                    min_detection_confidence=0.5, 
                                                    min_tracking_confidence=0.5)

    def process_frame(self, frame):
        """Process a frame and return face landmarks."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        return results.multi_face_landmarks

    def get_face_measurements(self, landmarks, frame_shape):
        """Extract detailed face measurements."""
        if not landmarks or len(landmarks) == 0:
            return None
        
        face_landmarks = landmarks[0].landmark
        def to_pixels(landmark):
            return [landmark.x * frame_shape[1], landmark.y * frame_shape[0]]

        # Key facial points
        forehead = to_pixels(face_landmarks[10])      # Top of forehead
        chin = to_pixels(face_landmarks[152])         # Bottom of chin
        left_cheek = to_pixels(face_landmarks[234])   # Left cheekbone
        right_cheek = to_pixels(face_landmarks[454])  # Right cheekbone
        left_jaw = to_pixels(face_landmarks[58])      # Left jawline
        right_jaw = to_pixels(face_landmarks[288])    # Right jawline

        # Measurements
        face_length = calculate_distance(forehead, chin)
        face_width = calculate_distance(left_cheek, right_cheek)
        jaw_width = calculate_distance(left_jaw, right_jaw)
        forehead_width = calculate_distance(to_pixels(face_landmarks[151]), 
                                            to_pixels(face_landmarks[9]))
        
        return {
            "face_length": face_length,
            "face_width": face_width,
            "jaw_width": jaw_width,
            "forehead_width": forehead_width,
            "face_ratio": face_width / face_length,
            "jaw_to_face_ratio": jaw_width / face_width,
            "forehead_to_face_ratio": forehead_width / face_width
        }

    def determine_face_shape(self, measurements):
        """Classify face shape with detailed criteria."""
        face_ratio = measurements["face_ratio"]
        jaw_to_face = measurements["jaw_to_face_ratio"]
        forehead_to_face = measurements["forehead_to_face_ratio"]

        if face_ratio >= 0.9 and face_ratio <= 1.1 and abs(jaw_to_face - forehead_to_face) < 0.1:
            return "Oval"  # Balanced width and length, rounded jaw
        elif face_ratio > 1.1 and jaw_to_face > 0.9:
            return "Square"  # Wide face, strong jawline
        elif face_ratio < 0.9 and jaw_to_face < 0.8:
            return "Heart"  # Narrower jaw, wider cheekbones
        elif face_ratio > 1.2 and forehead_to_face > jaw_to_face:
            return "Diamond"  # Wide cheekbones, narrower forehead and jaw
        elif face_ratio < 0.8:
            return "Oblong"  # Longer than wide, straight sides
        elif jaw_to_face > forehead_to_face and face_ratio < 1.0:
            return "Triangle"  # Wider jaw, narrower forehead
        else:
            return "Round"  # Nearly equal width and length, softer jaw

    def cleanup(self):
        """Release resources."""
        self.face_mesh.close()