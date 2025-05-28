#!/usr/bin/env python3
"""
Advanced Face Recognition Training
Captures faces from multiple angles and uses multiple detection methods
"""

import os
import cv2
import face_recognition
import dlib
import pickle
from pathlib import Path
import sys
import time
import platform
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODINGS_FILE, KNOWN_FACES_DIR

# Angle guidance messages
ANGLE_PROMPTS = [
    "Look straight at camera",
    "Turn head slightly LEFT",
    "Turn head slightly RIGHT", 
    "Tilt head UP slightly",
    "Tilt head DOWN slightly",
    "Turn head MORE to the LEFT (profile)",
    "Turn head MORE to the RIGHT (profile)",
    "Tilt head LEFT (ear to shoulder)",
    "Tilt head RIGHT (ear to shoulder)",
    "Look up-left corner",
    "Look up-right corner", 
    "Look down-left corner",
    "Look down-right corner"
]

def setup_detectors():
    """Setup multiple face detectors for better coverage"""
    detectors = {}
    
    # 1. Haar Cascade (works ok with angles)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detectors['haar'] = cv2.CascadeClassifier(cascade_path)
    
    # 2. Profile face cascade
    profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    detectors['profile'] = cv2.CascadeClassifier(profile_path)
    
    # 3. DNN face detector (more robust)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    proto_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if os.path.exists(proto_path) and os.path.exists(model_path):
        detectors['dnn'] = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    else:
        print("⚠ DNN model not found, will use other detectors")
        detectors['dnn'] = None
    
    return detectors

def detect_faces_multi(frame, detectors):
    """Detect faces using multiple methods and merge results"""
    h, w = frame.shape[:2]
    all_faces = []
    
    # Convert to grayscale for some detectors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Haar Cascade - Frontal
    if 'haar' in detectors and detectors['haar'] is not None:
        faces = detectors['haar'].detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            all_faces.append(('haar', x, y, w, h))
    
    # 2. Haar Cascade - Profile
    if 'profile' in detectors and detectors['profile'] is not None:
        # Check left profile
        faces = detectors['profile'].detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            all_faces.append(('profile-L', x, y, w, h))
        
        # Check right profile (flip image)
        flipped = cv2.flip(gray, 1)
        faces = detectors['profile'].detectMultiScale(flipped, 1.1, 3)
        for (x, y, w, h) in faces:
            # Adjust coordinates for flipped detection
            x_adj = frame.shape[1] - x - w
            all_faces.append(('profile-R', x_adj, y, w, h))
    
    # 3. DNN detector
    if 'dnn' in detectors and detectors['dnn'] is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        detectors['dnn'].setInput(blob)
        detections = detectors['dnn'].forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Lower threshold for training
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                all_faces.append(('dnn', x, y, x2-x, y2-y))
    
    # 4. Try face_recognition's own detector for difficult angles
    face_locations = face_recognition.face_locations(frame, model="cnn" if platform.system() != "Darwin" else "hog")
    for (top, right, bottom, left) in face_locations:
        all_faces.append(('fr', left, top, right-left, bottom-top))
    
    # Remove duplicates (overlapping detections)
    unique_faces = []
    for i, (method1, x1, y1, w1, h1) in enumerate(all_faces):
        is_duplicate = False
        for j, (method2, x2, y2, w2, h2) in enumerate(all_faces):
            if i != j and abs(x1-x2) < 50 and abs(y1-y2) < 50:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_faces.append((method1, x1, y1, w1, h1))
    
    return unique_faces

def capture_faces_advanced(name, camera_index=0):
    """Capture face images from multiple angles"""
    base_dir = Path(KNOWN_FACES_DIR)
    base_dir.mkdir(exist_ok=True)
    
    folder = base_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    
    # Setup detectors
    print("→ Setting up face detectors...")
    detectors = setup_detectors()
    
    # Open camera
    print(f"→ Opening camera (index {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Camera settings for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    time.sleep(2)
    
    # Test camera
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✗ Error: Cannot read from camera")
        cap.release()
        return None
    
    print("✓ Camera ready")
    
    print("\n=== Multi-Angle Capture Instructions ===")
    print("• We'll capture your face from multiple angles")
    print("• Follow the on-screen prompts for head positions")
    print("• SPACE: Capture photo for current angle")
    print("• N: Skip to next angle")
    print("• ESC: Finish capturing")
    print("• Try to keep your face in frame even at extreme angles")
    print("\nAim for at least 2-3 photos per angle\n")
    
    # Find starting index
    existing = list(folder.glob("*.jpg"))
    index = len(existing) + 1
    captured = 0
    angle_index = 0
    angle_captures = {}
    
    print("→ Camera window opening...")
    
    try:
        frame_count = 0
        while angle_index < len(ANGLE_PROMPTS):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Current angle prompt
            current_prompt = ANGLE_PROMPTS[angle_index]
            angle_captured = angle_captures.get(angle_index, 0)
            
            # Detect faces with multiple methods
            faces = detect_faces_multi(frame, detectors)
            
            # Draw all detected faces
            for (method, x, y, w, h) in faces:
                color = (0, 255, 0) if method == 'dnn' else (255, 255, 0) if 'profile' in method else (0, 255, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, method, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show instructions
            cv2.putText(display_frame, f"Angle {angle_index + 1}/{len(ANGLE_PROMPTS)}: {current_prompt}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Captured for this angle: {angle_captured}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Total captured: {captured}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "SPACE: Capture | N: Next angle | ESC: Finish", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Color indicator for number of captures at this angle
            indicator_color = (0, 255, 0) if angle_captured >= 3 else (0, 255, 255) if angle_captured >= 1 else (0, 0, 255)
            cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 20, indicator_color, -1)
            
            # Display frame
            cv2.imshow("Multi-Angle Face Capture", display_frame)
            
            # Make window come to front on first frame
            if frame_count == 1 and platform.system() == "Darwin":
                cv2.setWindowProperty("Multi-Angle Face Capture", cv2.WND_PROP_TOPMOST, 1)
            
            # Check for key press
            k = cv2.waitKey(1) & 0xFF
            
            if k == 27:  # ESC key
                print("→ Finishing capture...")
                break
            elif k == ord('n') or k == ord('N'):  # Next angle
                if angle_captured == 0:
                    print(f"⚠ No photos captured for: {current_prompt}")
                angle_index += 1
                print(f"→ Moving to next angle...")
            elif k == ord(' '):  # SPACE key
                if len(faces) == 0:
                    print("⚠ No face detected. Try adjusting your position.")
                else:
                    # Save the original frame (not display frame)
                    path = folder / f"{index}_angle{angle_index}.jpg"
                    cv2.imwrite(str(path), frame)
                    print(f"✓ Captured photo {index} - {current_prompt}")
                    index += 1
                    captured += 1
                    angle_captures[angle_index] = angle_captures.get(angle_index, 0) + 1
                    
                    # Auto-advance after 3 captures
                    if angle_captures[angle_index] >= 3:
                        angle_index += 1
                        if angle_index < len(ANGLE_PROMPTS):
                            print(f"→ Good coverage! Moving to: {ANGLE_PROMPTS[angle_index]}")
    
    except KeyboardInterrupt:
        print("\n→ Capture interrupted")
    finally:
        print("→ Closing camera...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    if captured == 0:
        print("⚠ No photos captured")
        return None
    
    print(f"\n✓ Captured {captured} photos for {name}")
    print(f"✓ Covered {len(angle_captures)} different angles")
    return folder

def encode_faces_robust():
    """Encode faces with better handling of difficult angles"""
    base_dir = Path(KNOWN_FACES_DIR)
    if not base_dir.exists():
        print(f"✗ Error: {KNOWN_FACES_DIR} directory not found")
        return False
    
    encodings, names = [], []
    total_faces = 0
    failed_faces = 0
    
    print("\n→ Encoding faces (this may take longer for profile views)...")
    
    for person_dir in sorted(base_dir.iterdir()):
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        person_encodings = 0
        
        for img_path in sorted(person_dir.glob("*.jpg")):
            try:
                # Load image
                img = face_recognition.load_image_file(str(img_path))
                
                # Try multiple detection models
                face_locations = face_recognition.face_locations(img, model="cnn" if platform.system() != "Darwin" else "hog")
                
                if not face_locations:
                    # Try with lower threshold
                    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=2)
                
                if not face_locations:
                    print(f"  ⚠ No face found in {img_path.name} (may be extreme angle)")
                    failed_faces += 1
                    continue
                
                if len(face_locations) > 1:
                    print(f"  ⚠ Multiple faces in {img_path.name}, using largest")
                    # Use the largest face
                    face_locations = [max(face_locations, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))]
                
                # Get encoding with more jitters for better accuracy
                enc = face_recognition.face_encodings(img, face_locations, num_jitters=2)
                if enc:
                    encodings.append(enc[0])
                    names.append(person_name)
                    person_encodings += 1
                    total_faces += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
                failed_faces += 1
        
        if person_encodings > 0:
            print(f"  ✓ {person_name}: {person_encodings} faces encoded")
        else:
            print(f"  ✗ {person_name}: No faces encoded")
    
    if not encodings:
        print("\n✗ No faces were successfully encoded")
        return False
    
    # Save encodings
    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump((encodings, names), f)
        print(f"\n✓ Saved {total_faces} face encodings to {ENCODINGS_FILE}")
        if failed_faces > 0:
            print(f"  ⚠ {failed_faces} images failed (likely extreme angles)")
            print(f"  This is normal - the system will still recognize you from captured angles")
        return True
    except Exception as e:
        print(f"\n✗ Error saving encodings: {e}")
        return False

def main():
    """Main training flow"""
    print("=== Advanced Face Recognition Training ===")
    print("This will capture your face from multiple angles for better recognition")
    
    # Test camera access
    print("→ Testing camera access...")
    camera_index = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            camera_index = idx
            cap.release()
            break
        cap.release()
    
    if camera_index is None:
        print("✗ No camera detected")
        return
    
    print(f"✓ Camera found at index {camera_index}")
    
    while True:
        # Get person name
        name = input("\nEnter person's name: ").strip()
        if not name:
            continue
        
        # Sanitize name
        for char in '<>:"|?*\\/.':
            name = name.replace(char, '_')
        
        # Capture faces from multiple angles
        folder = capture_faces_advanced(name, camera_index)
        
        if folder:
            response = input("\nAdd another person? (y/n): ").lower()
            if response != 'y':
                break
        else:
            response = input("\nTry again? (y/n): ").lower()
            if response != 'y':
                return
    
    # Encode all faces
    print("\n" + "="*30)
    success = encode_faces_robust()
    
    if success:
        print("\n✓ Advanced training complete!")
        print("  The system can now recognize you from various angles")
        print("  Note: Extreme profile views may still be challenging")
    else:
        print("\n✗ Training failed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n→ Training cancelled by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()