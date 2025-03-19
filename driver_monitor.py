import cv2
import dlib
import numpy as np
import pygame
import time
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DriverMonitor:
    def __init__(self):
        # Initialize face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Please check if your camera is connected and accessible.")
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
        except:
            print("Warning: Could not initialize audio. Alerts will be visual only.")
            self.alert_sound = None
            return
        
        # Constants for drowsiness detection
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 30
        self.frame_counter = 0
        
        # Constants for accident detection
        self.motion_threshold = 10000  # Increased threshold
        self.prev_frame = None
        self.accident_detected = False
        self.warmup_frames = 30  # Number of frames to wait before starting detection
        self.frame_count = 0
        
        # Alert settings
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds
        
        # Load alert sound
        try:
            self.alert_sound = pygame.mixer.Sound('alert.mp3')
        except:
            print("Warning: Could not load alert sound file. Alerts will be visual only.")
            self.alert_sound = None
    
    def calculate_ear(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR)"""
        # Compute the vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_drowsiness(self, frame, face):
        """Detect if the driver is drowsy"""
        # Get facial landmarks
        shape = self.predictor(frame, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Get eye landmarks
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Check if eyes are closed
        if ear < self.EYE_AR_THRESH:
            self.frame_counter += 1
            if self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
                return True
        else:
            self.frame_counter = 0
        return False
    
    def detect_accident(self, frame):
        """Detect potential accidents based on sudden movements"""
        # Skip detection during warmup period
        if self.frame_count < self.warmup_frames:
            self.frame_count += 1
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        
        # Calculate frame difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant movement
        total_movement = 0
        for contour in contours:
            total_movement += cv2.contourArea(contour)
        
        # Only trigger if total movement is significant
        if total_movement > self.motion_threshold:
            return True
        
        self.prev_frame = gray
        return False
    
    def send_alert(self, alert_type, message):
        """Send alerts through various channels"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        # Play sound alert
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except:
                print("Could not play sound alert")
        
        # Send HTTP alert (you can configure this to your preferred endpoint)
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Replace with your actual alert endpoint
            alert_url = os.getenv('ALERT_ENDPOINT', 'http://localhost:8000/alert')
            response = requests.post(alert_url, json=alert_data)
            print(f"Alert sent: {response.status_code}")
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
        
        self.last_alert_time = current_time
    
    def run(self):
        """Main monitoring loop"""
        print("Starting driver monitoring system...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detector(frame)
            
            for face in faces:
                # Draw face rectangle
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Check for drowsiness
                if self.detect_drowsiness(frame, face):
                    self.send_alert("drowsiness", "Driver appears to be drowsy!")
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check for accidents
            if self.detect_accident(frame):
                self.send_alert("accident", "Potential accident detected!")
                cv2.putText(frame, "ACCIDENT ALERT!", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Driver Monitor', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    monitor = DriverMonitor()
    monitor.run() 