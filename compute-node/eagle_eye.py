#!/usr/bin/env python3
"""
Simple Face + Object Detection
Uses MobileNet SSD for object detection (80 COCO classes)
"""

import cv2
import face_recognition
import pickle
import time
import sys
import os
import numpy as np
import urllib.request

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class SimpleObjectDetector:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.face_detector = None
        self.object_detector = None
        self.class_names = []
        
    def download_mobilenet_ssd(self):
        """Download MobileNet SSD model if not present"""
        model_path = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
        
        if not os.path.exists(model_path):
            print("→ Downloading MobileNet SSD model...")
            url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
            
            try:
                urllib.request.urlretrieve(url, model_path)
                print("✓ MobileNet SSD model downloaded")
            except Exception as e:
                print(f"✗ Failed to download MobileNet SSD model: {e}")
                return None
                
        return model_path
        
    def setup_detectors(self):
        """Setup OpenCV detectors"""
        print("→ Setting up detectors...")
        
        # Face detector (DNN)
        from config import MODELS_DIR
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        proto_path = os.path.join(MODELS_DIR, "deploy.prototxt")
        model_path = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if os.path.exists(proto_path) and os.path.exists(model_path):
            self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print("✓ Face detector ready")
        else:
            print("✗ DNN model files not found")
            print("  Please download:")
            print("  - deploy.prototxt")
            print("  - res10_300x300_ssd_iter_140000.caffemodel")
            print("  And place them in:", MODELS_DIR)
            self.face_detector = None
        
        # Object detector (YOLOv4-tiny)
        config_path = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
        weights_path = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
        
        if not os.path.exists(weights_path):
            print("⚠ YOLOv4-tiny model not found")
            return False
        
        # Load YOLOv4-tiny
        try:
            self.object_detector = cv2.dnn.readNet(weights_path, config_path)
            print("✓ YOLOv4-tiny object detector ready")
        except Exception as e:
            print(f"✗ Failed to load MobileNet SSD: {e}")
            return False
        
        # Load class names
        coco_path = os.path.join(MODELS_DIR, "coco.names")
        if os.path.exists(coco_path):
            with open(coco_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.class_names)} object classes")
        else:
            # MobileNet SSD default classes
            self.class_names = [
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"
            ]
            print("✓ Using default MobileNet SSD classes")
        
        return True
    
    def load_known_faces(self):
        """Load known face encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                self.known_encodings, self.known_names = pickle.load(f)
            print(f"✓ Loaded {len(self.known_encodings)} known faces")
            return True
        except:
            print("⚠ Face recognition data not found")
            return False
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN"""
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x and y2 > y:
                    faces.append((x, y, x2-x, y2-y))
        
        return faces
    
    def detect_objects_simple(self, frame):
        """Detect objects using YOLOv4-tiny"""
        if self.object_detector is None:
            return []
            
        objects = []
        h, w = frame.shape[:2]
        
        # Create blob from image (YOLO format)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Pass blob through network
        self.object_detector.setInput(blob)
        layer_outputs = self.object_detector.forward(self.object_detector.getUnconnectedOutLayersNames())
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Process YOLO outputs
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:  # Confidence threshold
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        # Color map for different object types
        color_map = {
            'person': [0, 255, 255],      # Yellow
            'car': [255, 0, 0],           # Blue
            'bicycle': [0, 255, 0],       # Green
            'motorbike': [255, 0, 255],   # Magenta
            'bus': [255, 128, 0],         # Orange
            'truck': [128, 0, 255],       # Purple
            'bottle': [0, 128, 255],      # Light blue
            'chair': [255, 255, 0],       # Cyan
            'dog': [128, 255, 0],         # Lime
            'cat': [255, 128, 128],       # Pink
        }
        
        # Process final detections
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # Get class name
                if 0 <= class_id < len(self.class_names):
                    label = self.class_names[class_id]
                    
                    # Skip background class
                    if label == "background":
                        continue
                    
                    # Ensure box is within frame
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(w, x + w_box)
                    y2 = min(h, y + h_box)
                    
                    if x2 > x and y2 > y:
                        # Get color for this object type
                        color = color_map.get(label, [255, 255, 255])  # Default white
                        
                        objects.append({
                            'bbox': (x, y, w_box, h_box),
                            'label': label,
                            'confidence': float(confidence),
                            'color': color
                        })
        
        return objects
    
    def recognize_faces(self, frame, face_locations):
        """Recognize detected faces"""
        if len(self.known_encodings) == 0 or len(face_locations) == 0:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to face_recognition format
        face_locs_fr = []
        for (x, y, w, h) in face_locations:
            face_locs_fr.append((y, x + w, y + h, x))
        
        # Get encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locs_fr)
        
        face_info = []
        for (x, y, w, h), encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.6)
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
    
    def draw_detections(self, frame, objects, faces):
        """Draw all detections"""
        # Count objects
        object_counts = {}
        
        # Draw objects
        for obj in objects:
            x, y, w, h = obj['bbox']
            label = obj['label']
            confidence = obj['confidence']
            
            # Count objects
            object_counts[label] = object_counts.get(label, 0) + 1
            
            # Draw box
            color = [int(c) for c in obj['color']]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label_text = f"{label} {confidence:.0%}"
            cv2.putText(frame, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw faces
        for face in faces:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            # Color based on recognition
            if name != "Unknown":
                color = (0, 255, 0)  # Green for known
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange for unknown
                thickness = 2
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw name
            if name != "Unknown":
                label = f"{name} ({confidence:.0%})"
            else:
                label = "Unknown"
            
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw summary
        y_pos = 30
        
        # Face count
        if len(faces) > 0:
            known = sum(1 for f in faces if f['name'] != "Unknown")
            unknown = sum(1 for f in faces if f['name'] == "Unknown")
            if known > 0:
                cv2.putText(frame, f"Known: {known}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
            if unknown > 0:
                cv2.putText(frame, f"Unknown: {unknown}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
        
        # Object counts
        for obj_type, count in sorted(object_counts.items()):
            cv2.putText(frame, f"{obj_type}: {count}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        
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
    print("=== MobileNet SSD Object Detection System ===")
    print("Detects: 80+ object classes from COCO dataset")
    print("Including: people, vehicles, animals, furniture, bottles, phones, etc.")
    
    # Initialize
    detector = SimpleObjectDetector()
    
    # Load faces
    detector.load_known_faces()
    
    # Setup detectors
    if not detector.setup_detectors():
        return
    
    # Connect
    cap = connect_to_stream()
    if cap is None:
        return
    
    print("\n→ Starting (press 'q' to quit)")
    print("Using MobileNet SSD for advanced object detection")
    
    # FPS
    fps_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            # Detect faces
            face_locations = detector.detect_faces_dnn(frame)
            
            # Recognize faces
            faces = detector.recognize_faces(frame, face_locations)
            
            # Detect objects using MobileNet SSD
            objects = detector.detect_objects_simple(frame)
            
            # Draw everything
            frame = detector.draw_detections(frame, objects, faces)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show
            cv2.imshow("MobileNet SSD Object Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n→ Stopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()