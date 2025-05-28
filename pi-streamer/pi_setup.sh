#!/bin/bash

# EyePi System - Raspberry Pi Setup
# Installs dependencies and configures camera streaming
# Supports both libcamera (new) and v4l2 (legacy) camera stacks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}==>${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Header
echo "======================================="
echo "EyePi - Pi Setup"
echo "======================================="
echo ""

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    print_success "Detected: $MODEL"
else
    print_warning "This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect OS version
print_status "Detecting OS version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_VERSION="$VERSION_CODENAME"
    print_success "OS: $PRETTY_NAME"
    
    # Check if using new camera stack (Bullseye and later)
    if [[ "$VERSION_ID" -ge "11" ]]; then
        CAMERA_STACK="libcamera"
        print_success "Will use libcamera (new camera stack)"
    else
        CAMERA_STACK="v4l2"
        print_success "Will use v4l2 (legacy camera stack)"
    fi
else
    CAMERA_STACK="auto"
    print_warning "Could not detect OS version, will auto-detect camera stack"
fi

# Update system
print_status "Updating package lists..."
sudo apt update

# Install FFmpeg
print_status "Installing FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    print_success "FFmpeg already installed"
else
    sudo apt install -y ffmpeg
    print_success "FFmpeg installed"
fi

# Install camera-specific packages
if [ "$CAMERA_STACK" = "libcamera" ]; then
    print_status "Installing libcamera tools..."
    if command -v libcamera-hello &> /dev/null; then
        print_success "libcamera already installed"
    else
        sudo apt install -y libcamera-tools
        print_success "libcamera-tools installed"
    fi
    
    # Test libcamera
    print_status "Testing libcamera..."
    if libcamera-hello --list-cameras &>/dev/null; then
        print_success "libcamera test successful"
        echo "Available cameras:"
        libcamera-hello --list-cameras
    else
        print_error "libcamera test failed"
        print_warning "You may need to enable the camera in raspi-config"
    fi
else
    # Install v4l-utils for legacy camera
    print_status "Installing camera utilities..."
    sudo apt install -y v4l-utils
    print_success "v4l-utils installed"
    
    # Enable camera module for legacy stack
    print_status "Checking camera module..."
    if [ -f /boot/config.txt ]; then
        if grep -q "^start_x=1" /boot/config.txt && grep -q "^gpu_mem=" /boot/config.txt; then
            print_success "Legacy camera already enabled"
        else
            print_warning "Enabling legacy camera module..."
            sudo raspi-config nonint do_camera 0
            print_warning "Reboot required for camera changes"
            REBOOT_REQUIRED=true
        fi
    fi
    
    # Load v4l2 module
    print_status "Loading camera module..."
    if lsmod | grep -q bcm2835_v4l2; then
        print_success "Camera module already loaded"
    else
        sudo modprobe bcm2835-v4l2
        print_success "Camera module loaded"
        echo "bcm2835-v4l2" | sudo tee -a /etc/modules
    fi
    
    # Check for camera devices
    print_status "Detecting camera devices..."
    echo ""
    v4l2-ctl --list-devices 2>/dev/null || print_warning "No camera devices found"
    echo ""
fi

# Get network information
print_status "Network information:"
IP_ADDR=$(hostname -I | awk '{print $1}') 
echo "  IP Address: $IP_ADDR"
echo "  Hostname: $(hostname)"
echo ""

# Create systemd service
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/face-recognition-stream.service > /dev/null << EOF
[Unit]
Description=Face Recognition Camera Stream
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="CAMERA_BACKEND=$CAMERA_STACK"
ExecStart=$(pwd)/pi_cam_stream.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service created"

# Create start/stop scripts
print_status "Creating helper scripts..."

# Start script
cat > start-stream.sh << 'EOF'
#!/bin/bash
echo "Starting face recognition stream..."
sudo systemctl start face-recognition-stream
sleep 2
sudo systemctl status face-recognition-stream --no-pager
EOF
chmod +x start-stream.sh

# Stop script
cat > stop-stream.sh << 'EOF'
#!/bin/bash
echo "Stopping face recognition stream..."
sudo systemctl stop face-recognition-stream
EOF
chmod +x stop-stream.sh

# Enable service script
cat > enable-autostart.sh << 'EOF'
#!/bin/bash
echo "Enabling auto-start on boot..."
sudo systemctl enable face-recognition-stream
echo "Stream will now start automatically on boot"
EOF
chmod +x enable-autostart.sh

# Logs script
cat > show-logs.sh << 'EOF'
#!/bin/bash
echo "Showing stream logs (Ctrl+C to exit)..."
sudo journalctl -u face-recognition-stream -f
EOF
chmod +x show-logs.sh

# Test camera script
cat > test-camera.sh << 'EOF'
#!/bin/bash
echo "Testing camera for 5 seconds..."
if command -v libcamera-hello &> /dev/null; then
    echo "Using libcamera..."
    libcamera-hello -t 5000
else
    echo "Using v4l2..."
    if [ -e /dev/video0 ]; then
        ffmpeg -f v4l2 -i /dev/video0 -t 5 -f null -
    else
        echo "No camera device found at /dev/video0"
    fi
fi
EOF
chmod +x test-camera.sh

print_success "Helper scripts created"

# Make pi_cam_stream.sh executable
chmod +x pi_cam_stream.sh

# Reload systemd
sudo systemctl daemon-reload

# Setup complete
echo ""
echo "======================================="
echo "Setup Complete!"
echo "======================================="
echo ""
echo "Your Pi's IP address: $IP_ADDR"
echo "Camera Stack: $CAMERA_STACK"
echo ""
echo "Available commands:"
echo "  ./test-camera.sh       - Test camera (5 sec preview)"
echo "  ./pi_cam_stream.sh     - Run stream manually"
echo "  ./start-stream.sh      - Start stream as service"
echo "  ./stop-stream.sh       - Stop stream service" 
echo "  ./show-logs.sh         - View stream logs"
echo "  ./enable-autostart.sh  - Enable auto-start on boot"
echo ""
echo "To start streaming:"
echo "  1. Test camera: ./test-camera.sh"
echo "  2. Run: ./pi_cam_stream.sh"
echo "  3. On your compute node, update config.py with:"
echo "     PI_IP = \"$IP_ADDR\""
echo ""
echo "Network flow: Pi ($IP_ADDR) → Compute Node"
echo ""

if [ "$REBOOT_REQUIRED" = true ]; then
    echo ""
    print_warning "IMPORTANT: Reboot required for camera module!"
    echo "Run: sudo reboot"
fi