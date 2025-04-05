import numpy as np
import cv2

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def display_text(image, text, position, font_scale=0.7, color=(255, 255, 0), thickness=2):
    """Display text on an image."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)