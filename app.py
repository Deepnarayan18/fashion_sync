from flask import Flask, request, render_template, jsonify, Response
import cv2
import numpy as np
from posestimation import PoseEstimator
from face_analysis import FaceAnalyzer
from fashion_advice import FashionAdvisor
import mediapipe as mp

app = Flask(__name__)

# Initialize components
pose_estimator = PoseEstimator()
face_analyzer = FaceAnalyzer()
fashion_advisor = FashionAdvisor()
camera = None  # Camera starts as None, initialized only when started

def estimate_gender(body_measurements):
    """Simple heuristic for gender estimation based on shoulder-to-hip ratio."""
    if body_measurements:
        shoulder_to_hip = body_measurements["shoulder_to_hip_ratio"]
        if shoulder_to_hip > 1.2:
            return "Men"
        elif shoulder_to_hip < 1.1:
            return "Woman"
        else:
            return "Uncertain"
    return "Unknown"

def process_frame_with_overlays(frame, pose_landmarks, face_landmarks):
    """Process a frame and add detection overlays with high accuracy."""
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    body_measurements = None
    if pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_drawing.draw_landmarks(
            processed_frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        body_measurements = pose_estimator.get_measurements(pose_landmarks, processed_frame.shape)

    # Draw face landmarks
    face_measurements = None
    if face_landmarks and len(face_landmarks) > 0:
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh  # Corrected from ENGface_mesh
        for face_lms in face_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame, face_lms, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            )
        face_measurements = face_analyzer.get_face_measurements(face_landmarks, processed_frame.shape)

    # Overlay text
    font_scale = 0.7
    color = (255, 255, 255)  # White text
    thickness = 2
    y_offset = 30

    if body_measurements:
        body_type = pose_estimator.determine_body_type(body_measurements)
        gender = estimate_gender(body_measurements)
        cv2.putText(processed_frame, f"Body Type: {body_type}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(processed_frame, f"Gender: {gender}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(processed_frame, f"Shoulder: {body_measurements['shoulder_width']:.1f}px", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(processed_frame, f"Torso: {body_measurements['torso_length']:.1f}px", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(processed_frame, f"Hip: {body_measurements['hip_width']:.1f}px", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += 30
        leg_length = (body_measurements['left_leg_length'] + body_measurements['right_leg_length']) / 2
        cv2.putText(processed_frame, f"Leg: {leg_length:.1f}px", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if face_measurements:
        face_shape = face_analyzer.determine_face_shape(face_measurements)
        y_offset += 30
        cv2.putText(processed_frame, f"Face Shape: {face_shape}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return processed_frame, body_measurements, face_measurements

def generate_camera_frames():
    """Generate frames from the camera with overlays."""
    global camera
    if camera is None or not camera.isOpened():
        return
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        pose_landmarks = pose_estimator.process_frame(frame)
        face_landmarks = face_analyzer.process_frame(frame)
        processed_frame, _, _ = process_frame_with_overlays(frame, pose_landmarks, face_landmarks)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return jsonify({"status": "Camera started"})

@app.route('/video_feed')
def video_feed():
    """Stream live camera feed with detection and overlays."""
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process', methods=['POST'])
def process_input():
    """Process the current frame and get fashion advice."""
    global camera
    if camera is None or not camera.isOpened():
        return jsonify({"error": "Camera is not started"}), 400

    season = request.form.get('season', 'Winter')
    ret, frame = camera.read()
    if not ret:
        return jsonify({"error": "Camera not accessible"}), 500

    pose_landmarks = pose_estimator.process_frame(frame)
    face_landmarks = face_analyzer.process_frame(frame)
    processed_frame, body_measurements, face_measurements = process_frame_with_overlays(frame, pose_landmarks, face_landmarks)

    if body_measurements and face_measurements:
        body_type = pose_estimator.determine_body_type(body_measurements)
        face_shape = face_analyzer.determine_face_shape(face_measurements)
        gender = estimate_gender(body_measurements)
        advice = fashion_advisor.get_recommendation(body_measurements, body_type, face_measurements, face_shape, season, gender)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = buffer.tobytes().hex()
        return jsonify({
            "body_type": body_type,
            "face_shape": face_shape,
            "gender": gender,
            "measurements": {
                "shoulder_width": f"{body_measurements['shoulder_width']:.1f}px",
                "torso_length": f"{body_measurements['torso_length']:.1f}px",
                "hip_width": f"{body_measurements['hip_width']:.1f}px",
                "leg_length": f"{(body_measurements['left_leg_length'] + body_measurements['right_leg_length'])/2:.1f}px"
            },
            "advice": advice,
            "image": img_base64
        })
    else:
        return jsonify({"error": "Could not detect pose or face"}), 400

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop and release the camera resource."""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
    return jsonify({"status": "Camera stopped"})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if camera is not None and camera.isOpened():
            camera.release()