from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class FashionAdvisor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
        self.client = Groq(api_key=api_key)

    def get_recommendation(self, body_measurements, body_type, face_measurements, face_shape, season="Winter", gender="Uncertain"):
        """Generate detailed, gender-specific fashion advice with bold points and icons."""
        prompt = (
            f"I am a {gender} with a {body_type} body type. My measurements are: shoulder width {body_measurements['shoulder_width']:.1f}px, "
            f"torso length {body_measurements['torso_length']:.1f}px, hip width {body_measurements['hip_width']:.1f}px, "
            f"waist width {body_measurements['waist_width']:.1f}px, left arm length {body_measurements['left_arm_length']:.1f}px, "
            f"right arm length {body_measurements['right_arm_length']:.1f}px, left leg length {body_measurements['left_leg_length']:.1f}px, "
            f"right leg length {body_measurements['right_leg_length']:.1f}px, shoulder-to-hip ratio {body_measurements['shoulder_to_hip_ratio']:.2f}, "
            f"torso-to-leg ratio {body_measurements['torso_to_leg_ratio']:.2f}. "
            f"My face is {face_shape} with face length {face_measurements['face_length']:.1f}px, face width {face_measurements['face_width']:.1f}px, "
            f"jaw width {face_measurements['jaw_width']:.1f}px, forehead width {face_measurements['forehead_width']:.1f}px, "
            f"face ratio {face_measurements['face_ratio']:.2f}. "
            f"Provide detailed fashion advice for {season}, tailored to my gender, body type, and face shape. "
            f"Include specific recommendations under the following bold headings: **Clothing**, **Accessories**, **Hairstyle**. "
            f"Add relatable emoji icons (e.g., 👕 for clothing, 🧢 for accessories, ✂️ for hairstyle) before each heading. "
            f"Ensure suggestions are complete, informative, and practical."
        )

        response = self.client.chat.completions.create(
            model="qwen-2.5-coder-32b",
            messages=[
                {"role": "system", "content": "You are a fashion expert providing detailed, personalized advice with structured formatting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10200,
            temperature=0.7
        )
        return response.choices[0].message.content