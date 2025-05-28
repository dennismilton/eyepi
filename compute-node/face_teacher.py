#!/usr/bin/env python3
"""
Face Recognition System - Training
CLI-based face registration using webcam
"""

import os
import cv2
import face_recognition
import pickle
from pathlib import Path
import sys
import time
import platform

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODINGS_FILE, KNOWN_FACES_DIR

def test_camera_access():
    """Test camera access with multiple attempts"""
    print("→ Testing camera access...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"  Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        # Give camera time to initialize
        time.sleep(0.5)
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ Camera found at index {camera_index}")
            cap.release()
            return camera_index
        
        cap.release()
    
    return None

def validate_camera():
    """Check if camera is available with better error handling"""
    # Check if running on macOS
    if platform.system() == "Darwin":
        print("→ Running on macOS - checking camera permissions...")
        print("  If prompted, please grant camera access to Terminal/Python")
        time.sleep(1)
    
    # Test camera access
    camera_index = test_camera_access()
    
    if camera_index is None:
        print("\n✗ Error: No camera detected.")
        print("\nTroubleshooting steps:")
        print("1. macOS Camera Permissions:")
        print("   - Go to System Settings > Privacy & Security > Camera")
        print("   - Enable camera access for Terminal")
        print("   - You may need to restart Terminal after granting permission")
        print("\n2. Check if another app is using the camera:")
        print("   - Close Zoom, Teams, FaceTime, etc.")
        print("\n3. Try running with sudo (not recommended but can help diagnose):")
        print("   - sudo python compute-node/face_teacher.py")
        print("\n4. Reset camera permissions:")
        print("   - Run: tccutil reset Camera")
        print("   - Then re-run this script and grant permission when prompted")
        sys.exit(1)
    
    print("✓ Camera validated and ready")
    return camera_index

def get_person_name():
    """Get and validate person's name"""
    while True:
        name = input("\nEnter person's name: ").strip()
        if not name:
            print("⚠ Name cannot be empty. Please try again.")
            continue
        
        # Sanitize name for filesystem
        invalid_chars = '<>:"|?*\\/.'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        return name

def capture_faces(name, camera_index=0):
    """Capture face images from webcam"""
    base_dir = Path(KNOWN_FACES_DIR)
    base_dir.mkdir(exist_ok=True)
    
    folder = base_dir / name
    
    # Check if person already exists
    if folder.exists() and any(folder.glob("*.jpg")):
        response = input(f"⚠ {name} already has images. Add more? (y/n): ").lower()
        if response != 'y':
            return None
    
    folder.mkdir(parents=True, exist_ok=True)
    
    # Open camera with specific index
    print(f"→ Opening camera (index {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties for better compatibility
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Wait for camera to warm up
    print("→ Waiting for camera to initialize...")
    time.sleep(2)
    
    # Test camera
    ret, frame = cap.read()
    if not ret or frame is None:
        print("✗ Error: Cannot read from camera")
        print("  The camera was detected but cannot capture frames.")
        print("  Try closing other applications using the camera.")
        cap.release()
        return None
    
    print("✓ Camera ready")
    
    print("\n=== Capture Instructions ===")
    print("• SPACE: Capture photo")
    print("• ESC: Finish capturing")
    print("• Move your head slightly between captures")
    print("• Aim for 5-10 photos\n")
    
    # Find starting index
    existing = list(folder.glob("*.jpg"))
    index = len(existing) + 1
    captured = 0
    
    # Setup face detection for preview
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Check if cascade loaded
    if face_cascade.empty():
        print("⚠ Warning: Face detection cascade not loaded")
        print("  Face preview boxes won't be shown, but capture will still work")
    
    print("→ Camera window opening...")
    print("  If you don't see the camera window, check behind other windows")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠ Lost camera connection, retrying...")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Show face detection preview if cascade loaded
            if not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                faces = []  # Assume face is present if cascade not loaded
            
            # Show capture count and instructions
            cv2.putText(display_frame, f"Captured: {captured} photos", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press SPACE to capture, ESC to finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Face Capture", display_frame)
            
            # Make window come to front (macOS)
            if frame_count == 1 and platform.system() == "Darwin":
                cv2.setWindowProperty("Face Capture", cv2.WND_PROP_TOPMOST, 1)
            
            # Check for key press
            k = cv2.waitKey(1) & 0xFF
            
            if k == 27:  # ESC key
                print("→ Finishing capture...")
                break
            elif k == ord(' '):  # SPACE key
                # If cascade not loaded, always allow capture
                if face_cascade.empty() or len(faces) == 1:
                    path = folder / f"{index}.jpg"
                    # Save the flipped frame
                    cv2.imwrite(str(path), frame)
                    print(f"✓ Captured photo {index}")
                    index += 1
                    captured += 1
                elif len(faces) == 0:
                    print("⚠ No face detected. Please position your face in view.")
                elif len(faces) > 1:
                    print("⚠ Multiple faces detected. Please ensure only one person is visible.")
    
    except KeyboardInterrupt:
        print("\n→ Capture interrupted")
    finally:
        print("→ Closing camera...")
        cap.release()
        cv2.destroyAllWindows()
        # Give time for window to close
        cv2.waitKey(1)
    
    if captured == 0:
        print("⚠ No photos captured")
        return None
    
    print(f"\n✓ Captured {captured} photos for {name}")
    return folder

def encode_faces():
    """Encode all faces in the known_faces directory"""
    base_dir = Path(KNOWN_FACES_DIR)
    if not base_dir.exists():
        print(f"✗ Error: {KNOWN_FACES_DIR} directory not found")
        return False
    
    encodings, names = [], []
    total_faces = 0
    failed_faces = 0
    
    print("\n→ Encoding faces...")
    
    for person_dir in sorted(base_dir.iterdir()):
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        person_encodings = 0
        
        for img_path in sorted(person_dir.glob("*.jpg")):
            try:
                # Load and encode image
                img = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(img)
                
                if not face_locations:
                    print(f"  ⚠ No face found in {img_path.name}")
                    failed_faces += 1
                    continue
                
                if len(face_locations) > 1:
                    print(f"  ⚠ Multiple faces in {img_path.name}, using first")
                
                enc = face_recognition.face_encodings(img, face_locations[:1])
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
            print(f"  ⚠ {failed_faces} images failed to encode")
        return True
    except Exception as e:
        print(f"\n✗ Error saving encodings: {e}")
        return False

def main():
    """Main training flow"""
    print("=== Face Recognition Training ===")
    
    # Validate camera and get working index
    camera_index = validate_camera()
    
    while True:
        # Get person name
        name = get_person_name()
        
        # Capture faces
        folder = capture_faces(name, camera_index)
        
        if folder:
            # Ask if user wants to add more people
            response = input("\nAdd another person? (y/n): ").lower()
            if response != 'y':
                break
        else:
            response = input("\nTry again? (y/n): ").lower()
            if response != 'y':
                print("→ Exiting without encoding")
                return
    
    # Encode all faces
    print("\n" + "="*30)
    success = encode_faces()
    
    if success:
        print("\n✓ Training complete!")
        print(f"  Face database saved to: {ENCODINGS_FILE}")
        print("  You can now run recognizer.py")
    else:
        print("\n✗ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n→ Training cancelled by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)