# EyePi - Distributed Face Recognition System

A high-performance distributed face recognition system designed for maximum efficiency:
- **Raspberry Pi**: Streams camera feed via RTSP (lightweight)
- **Compute Node**: Receives stream and performs face recognition (compute-heavy)
- **Training**: CLI interface to register new faces using local webcam

## System Architecture

```
Raspberry Pi (Stream Source)          Compute Node (192.168.1.134)
┌─────────────────────┐              ┌──────────────────────────┐
│ Camera → RTSP Server│ ──Network──> │ RTSP Client → Recognition│
└─────────────────────┘              │         ↓                │
                                     │ [Face Encodings DB]      │
                                     └──────────────────────────┘
```

- Pi captures video and serves RTSP stream
- Compute node receives stream and processes faces
- All heavy computation happens on the compute node

## Components

### 1. Setup Scripts
- **`setup.sh`**: Compute node automated setup with venv creation
- **`pi-streamer/pi_setup.sh`**: Pi automated setup with systemd service
- Both scripts include dependency installation and validation

### 2. Configuration (`config.py`)
- Central configuration for all settings
- IP addresses, detection thresholds, file paths
- Import with: `from config import *`

### 3. Pi Camera Streamer (`pi-streamer/pi_cam_stream.sh`)
- **Direct TCP streaming** via FFmpeg (no MediaMTX dependency)
- Auto-detects camera backend (libcamera vs v4l2)
- Supports modern libcamera stack (Bullseye+)
- Falls back to legacy v4l2 for older OS
- Configurable via environment variables
- Systemd service support via pi_setup.sh

### 4. Face Training
- **`compute-node/face_teacher.py`**: Basic training with webcam
- **`compute-node/angle_master.py`**: Advanced multi-angle training
- Interactive CLI with face detection preview
- Validates camera availability
- Handles existing person updates
- Comprehensive error reporting
- Batch encoding of all faces

### 5. Face Recognition
- **`compute-node/face_spy.py`**: Clean recognition with duplicate suppression via NMS
- **`compute-node/eagle_eye.py`**: Combined face + object detection (YOLOv4-tiny)
- All versions feature:
  - **TCP-primary streaming** with RTSP fallback
  - DNN face detection with higher confidence thresholds
  - Non-maximum suppression to prevent duplicates
  - Confidence scores on matches
  - FPS display and performance monitoring
  - Automatic reconnection on stream loss

## Key Features

### Error Handling
- Camera validation before use
- RTSP connection retry with fallback
- Graceful handling of missing files
- Face detection validation during capture
- Boundary checking for face regions

### Performance Optimizations
1. **Distributed Processing**: Offload compute to dedicated node
2. **Direct TCP Streaming**: Simple FFmpeg streaming (no MediaMTX overhead)
3. **Two-Stage Detection**: DNN detection → face_recognition encoding
4. **Pre-computed Encodings**: One-time encoding during training
5. **Low Resolution Stream**: 640x480 @ 15 FPS default
6. **Region Processing**: Only encode detected faces
7. **Buffer Size**: Reduced latency with CAP_PROP_BUFFERSIZE=1
8. **Non-Maximum Suppression**: Removes duplicate detections

### User Experience
- Clear status messages with symbols (✓ ✗ → ⚠)
- Face detection preview during training
- Capture counter and instructions
- FPS display during recognition
- Detailed error messages with solutions

## Common Commands

```bash
# Initial setup (one-time)
./setup.sh

# Activate virtual environment (every session)
source venv/bin/activate

# Configure system
# Edit config.py - set PI_IP

# Train faces
python compute-node/face_teacher.py     # Basic training
python compute-node/angle_master.py     # Multi-angle training

# Run recognition
python compute-node/face_spy.py         # Clean face recognition
python compute-node/eagle_eye.py        # Face + object detection

# Stream from Pi
cd pi-streamer && ./pi_cam_stream.sh

# Deactivate venv when done
deactivate
```

## File Structure

All training data and models are stored in the `compute-node/` directory:

```
compute-node/
├── face_encodings.pkl       # Generated face database
├── known_faces/            # Generated training images
│   ├── person1/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── person2/
│       └── ...
├── models/                 # Downloaded detection models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   └── coco.names
└── [recognition scripts]
```

## Development Tips

- Test with local webcam first (set RTSP_URL to 0)
- Monitor network latency between Pi and compute node
- Use `cv2.CAP_PROP_FPS` to check actual FPS
- Profile with `cProfile` for bottlenecks
- Consider batch processing for multiple faces

## Current Architecture Notes

- **TCP-first streaming**: Direct FFmpeg TCP streaming (primary) with RTSP fallback
- **No MediaMTX dependency**: Simplified setup using FFmpeg's built-in TCP server
- Face detection uses OpenCV DNN models with higher confidence thresholds (0.7)
- YOLOv4-tiny provides object detection capabilities in eagle_eye.py
- Non-maximum suppression prevents duplicate detections in face_spy.py
- Multi-angle training improves recognition robustness
- Performance trade-off: Slightly slower (~15-25ms/frame) for better accuracy

## Future Enhancements

Potential improvements identified:
- SQLite database instead of pickle for face storage
- Flask dashboard for remote monitoring
- Real-time analytics and performance metrics
- Multi-camera support with load balancing
- Mobile app integration
- Emotion detection and analytics
- Face clustering for unknown persons
- Liveness detection (anti-spoofing)