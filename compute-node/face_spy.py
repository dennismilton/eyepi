#!/usr/bin/env python3
"""
Clean Face Recognition - No Duplicate Detections
Single detection per face, cleaner output
"""

import cv2
import face_recognition
import pickle
import time
import sys
import os
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class CleanRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.face_detector = None
        
    def load_known_faces(self):
        """Load known face encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                self.known_encodings, self.known_names = pickle.load(f)
            print(f"✓ Loaded {len(self.known_encodings)} known faces")
            return True
        except FileNotFoundError:
            print(f"✗ Error: {ENCODINGS_FILE} not found.")
            print("  Run train.py first to create face encodings.")
            return False
        except Exception as e:
            print(f"✗ Error loading encodings: {e}")
            return False
    
    def setup_face_detector(self):
        """Setup DNN face detector"""
        print("→ Setting up face detector...")
        
        proto_path = os.path.join(MODELS_DIR, "deploy.prototxt")
        model_path = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if os.path.exists(proto_path) and os.path.exists(model_path):
            self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print("✓ Face detector ready")
            return True
        else:
            print("✗ DNN model files not found")
            print("  Please download:")
            print("  - deploy.prototxt")
            print("  - res10_300x300_ssd_iter_140000.caffemodel")
            print("  And place them in:", MODELS_DIR)
            return False
    
    def detect_faces_clean(self, frame):
        """Detect faces with non-maximum suppression to avoid duplicates"""
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Collect all detections
        boxes = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Higher threshold for cleaner detection
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x and y2 > y:
                    boxes.append([x, y, x2 - x, y2 - y])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression to remove overlapping boxes
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)
            clean_faces = []
            if len(indices) > 0:
                for i in indices.flatten():
                    clean_faces.append(tuple(boxes[i]))
            return clean_faces
        
        return []
    
    def recognize_faces(self, frame, face_locations):
        """Recognize detected faces"""
        if len(self.known_encodings) == 0 or len(face_locations) == 0:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to face_recognition format: (top, right, bottom, left)
        face_locs_fr = []
        for (x, y, w, h) in face_locations:
            face_locs_fr.append((y, x + w, y + h, x))
        
        # Get encodings
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locs_fr, num_jitters=1)
        except:
            return []
        
        face_info = []
        for (x, y, w, h), encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                encoding, 
                tolerance=FACE_RECOGNITION_TOLERANCE
            )
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = self.known_names[best_match_idx]
                    confidence = 1 - face_distances[best_match_idx]
            
            face_info.append({
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })
        
        return face_info
    
    def process_frame(self, frame):
        """Process frame with clean face detection"""
        # Detect faces (no duplicates)
        face_locations = self.detect_faces_clean(frame)
        
        # Recognize faces
        faces = self.recognize_faces(frame, face_locations)
        
        # Draw results
        for face in faces:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            # Color based on recognition
            if name != "Unknown":
                color = (0, 255, 0)  # Green for known faces
                thickness = 3
                label = f"{name} ({confidence:.0%})"
            else:
                color = (0, 165, 255)  # Orange for unknown faces
                thickness = 2
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = y - 10 if y > 30 else y + h + 25
            
            # Background rectangle for text
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0], label_y + 5), color, -1)
            
            # Text
            cv2.putText(frame, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame

def connect_to_stream():
    """Connect to stream"""
    print(f"→ Connecting to TCP stream: {TCP_URL}")
    cap = cv2.VideoCapture(TCP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"→ TCP failed, trying RTSP: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("✗ Failed to connect")
        return None
    
    print("✓ Connected")
    return cap

def main():
    """Main loop"""
    print("=== Clean Face Recognition System ===")
    print("Single detection per face, no duplicates")
    
    # Initialize
    recognizer = CleanRecognizer()
    
    # Load faces
    if not recognizer.load_known_faces():
        return
    
    # Setup detector
    if not recognizer.setup_face_detector():
        return
    
    # Connect
    cap = connect_to_stream()
    if cap is None:
        return
    
    print("\n→ Starting clean recognition (press 'q' to quit)")
    
    # FPS calculation
    fps_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            # Process frame
            frame = recognizer.process_frame(frame)
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Clean Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n→ Stopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()