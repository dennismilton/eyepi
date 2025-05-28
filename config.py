# Configuration file for EyePi System

# Stream Settings
# The compute node will receive the stream from the Pi
# Replace PI_IP with your Raspberry Pi's actual IP address
PI_IP = "192.168.1.127"  # Replace with your Pi's IP address (e.g., "192.168.1.100")
RTSP_PORT = 8554
RTSP_PATH = "live.stream"
TCP_PORT = 8554

# Stream URLs - TCP is primary, RTSP is fallback
TCP_URL = f"tcp://{PI_IP}:{TCP_PORT}"
RTSP_URL = f"rtsp://{PI_IP}:{RTSP_PORT}/{RTSP_PATH}"

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Face Detection Settings
MIN_DETECTION_CONFIDENCE = 0.6
FACE_RECOGNITION_TOLERANCE = 0.6  # Lower = stricter matching

# File Paths - Always store in compute-node directory
import os
_BASE_DIR = os.path.join(os.path.dirname(__file__), "compute-node")
ENCODINGS_FILE = os.path.join(_BASE_DIR, "face_encodings.pkl")
KNOWN_FACES_DIR = os.path.join(_BASE_DIR, "known_faces")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

# Display Settings
FONT_SCALE = 0.9
FONT_COLOR = (0, 255, 0)  # Green
BOX_COLOR = (0, 255, 0)   # Green
BOX_THICKNESS = 2
