#!/bin/bash

# Face Recognition System - Pi Camera TCP Streamer
# Streams directly via TCP (simple and reliable)

# Default configuration
WIDTH=${CAMERA_WIDTH:-640}
HEIGHT=${CAMERA_HEIGHT:-480}
FPS=${CAMERA_FPS:-15}
TCP_PORT=${TCP_PORT:-8554}

# Camera backend - auto-detect or override with CAMERA_BACKEND env var
CAMERA_BACKEND=${CAMERA_BACKEND:-auto}

# Get Pi's IP address
PI_IP=$(hostname -I | awk '{print $1}')
if [ -z "$PI_IP" ]; then
    echo "Error: Could not determine IP address"
    exit 1
fi

# Function to check if libcamera is available
check_libcamera() {
    if command -v libcamera-vid &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if v4l2 device exists
check_v4l2() {
    if [ -e "/dev/video0" ]; then
        return 0
    else
        return 1
    fi
}

# Auto-detect camera backend if needed
if [ "$CAMERA_BACKEND" = "auto" ]; then
    if check_libcamera; then
        CAMERA_BACKEND="libcamera"
        echo "Auto-detected: libcamera (new camera stack)"
    elif check_v4l2; then
        CAMERA_BACKEND="v4l2"
        echo "Auto-detected: v4l2 (legacy camera stack)"
    else
        echo "Error: No camera backend detected"
        echo "Please check:"
        echo "  1. Camera is connected properly"
        echo "  2. Camera is enabled in raspi-config"
        echo "  3. For new OS: libcamera-tools is installed"
        echo "  4. For legacy: bcm2835-v4l2 module is loaded"
        exit 1
    fi
fi

# Display configuration
echo "=== Pi Camera TCP Streamer ==="
echo "Camera Backend: $CAMERA_BACKEND"
echo "Resolution: ${WIDTH}x${HEIGHT} @ ${FPS}fps"
echo "TCP Stream: tcp://${PI_IP}:${TCP_PORT}"
echo ""
echo "Streaming directly via TCP (simple and reliable)"
echo "Your compute node should connect to: tcp://${PI_IP}:${TCP_PORT}"
echo ""
echo "Make sure config.py on your compute node has:"
echo "  PI_IP = \"${PI_IP}\""
echo ""
echo "Starting stream... (Ctrl+C to stop)"

# Start streaming based on backend
if [ "$CAMERA_BACKEND" = "libcamera" ]; then
    echo "→ Streaming with libcamera (TCP)..."
    libcamera-vid \
        --width $WIDTH \
        --height $HEIGHT \
        --framerate $FPS \
        --codec h264 \
        --profile baseline \
        --level 4.1 \
        --inline \
        --nopreview \
        -t 0 \
        -o - | \
    ffmpeg -f h264 \
           -i - \
           -c:v copy \
           -f mpegts \
           -listen 1 \
           tcp://0.0.0.0:${TCP_PORT}
           
elif [ "$CAMERA_BACKEND" = "v4l2" ]; then
    # Legacy v4l2 TCP streaming
    DEVICE=${CAMERA_DEVICE:-/dev/video0}
    
    # Check if device exists
    if [ ! -e "$DEVICE" ]; then
        echo "Error: Camera device $DEVICE not found"
        echo "Available devices:"
        ls /dev/video* 2>/dev/null || echo "No video devices found"
        exit 1
    fi
    
    echo "→ Streaming with v4l2 (TCP)..."
    ffmpeg -f v4l2 \
           -framerate $FPS \
           -video_size ${WIDTH}x${HEIGHT} \
           -i $DEVICE \
           -c:v libx264 \
           -preset ultrafast \
           -tune zerolatency \
           -b:v 2M \
           -f mpegts \
           -listen 1 \
           tcp://0.0.0.0:${TCP_PORT}
else
    echo "Error: Unknown camera backend: $CAMERA_BACKEND"
    echo "Valid options: libcamera, v4l2, auto"
    exit 1
fi