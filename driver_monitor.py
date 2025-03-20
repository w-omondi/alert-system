import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from shared import alert_queue

# Load environment variables
load_dotenv()

class DriverMonitor:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera with better error handling
        self.cap = None
        for i in range(2):  # Try first two camera indices
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = self.cap.read()
                if ret:
                    print(f"Successfully initialized camera {i}")
                    break
                else:
                    print(f"Camera {i} opened but couldn't read frame")
                    self.cap.release()
                    self.cap = None
            else:
                print(f"Could not open camera {i}")
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("""
                Could not initialize camera. Please check:
                1. Is your camera connected?
                2. Do you have permission to access the camera?
                3. Is another application using the camera?
                4. Try running with sudo or adding your user to the video group:
                   sudo usermod -a -G video $USER
                   (Then log out and back in)
            """)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
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
        self.motion_threshold = 10000
        self.prev_frame = None
        self.accident_detected = False
        self.warmup_frames = 30
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
        """Calculate the Eye Aspect Ratio (EAR) using MediaPipe landmarks"""
        # Get the eye landmarks (6 points for each eye)
        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        
        # Calculate vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Calculate horizontal distance
        h = np.linalg.norm(p1 - p4)
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_drowsiness(self, frame, face_landmarks):
        """Detect if the driver is drowsy using MediaPipe landmarks"""
        if not face_landmarks:
            return False
            
        # Get eye landmarks (MediaPipe indices for 6 points per eye)
        left_eye = np.array([face_landmarks.landmark[i] for i in [33, 246, 161, 160, 159, 158]])
        right_eye = np.array([face_landmarks.landmark[i] for i in [362, 398, 384, 385, 386, 387]])
        
        # Convert to numpy arrays and scale to frame size
        h, w = frame.shape[:2]
        left_eye = np.array([[int(p.x * w), int(p.y * h)] for p in left_eye])
        right_eye = np.array([[int(p.x * w), int(p.y * h)] for p in right_eye])
        
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
        
        # Add alert to queue
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        alert_queue.put(alert_data)
        
        # Send HTTP alert
        try:
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
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh using the correct method
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Check for drowsiness
                    if self.detect_drowsiness(frame, face_landmarks):
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