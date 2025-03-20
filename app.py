from flask import Flask, render_template, Response
from driver_monitor import DriverMonitor
import threading
import queue
from shared import alert_queue
import cv2
import time
import mediapipe as mp

app = Flask(__name__)

# Global variables for video streaming
camera = None
frame_queue = queue.Queue(maxsize=10)

def generate_frames():
    """Generate frames from the camera"""
    global camera
    while True:
        if camera is None:
            try:
                camera = DriverMonitor()
            except Exception as e:
                print(f"Failed to initialize camera: {str(e)}")
                time.sleep(1)
                continue

        ret, frame = camera.cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = camera.face_mesh.process(rgb_frame)
        
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
                if camera.detect_drowsiness(frame, face_landmarks):
                    camera.send_alert("drowsiness", "Driver appears to be drowsy!")
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check for accidents
        if camera.detect_accident(frame):
            camera.send_alert("accident", "Potential accident detected!")
            cv2.putText(frame, "ACCIDENT ALERT!", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    """Get alerts from the queue"""
    alerts = []
    while not alert_queue.empty():
        alerts.append(alert_queue.get())
    return {'alerts': alerts}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 