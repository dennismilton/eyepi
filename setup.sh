#!/bin/bash

# EyePi System - Setup Script
# Creates virtual environment and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
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
echo "==================================="
echo "EyePi System Setup"
echo "==================================="
echo ""

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python $PYTHON_VERSION found"
        
        # Check for Python 3.13 (MediaPipe compatibility issue)
        if [ "$PYTHON_MINOR" -ge 13 ]; then
            print_warning "Python 3.13+ detected - MediaPipe may not be available"
            print_status "Will use alternative face detection method"
            USE_MEDIAPIPE=false
        else
            USE_MEDIAPIPE=true
        fi
    else
        print_error "Python 3.8+ required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check if venv already exists
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
    else
        print_status "Using existing virtual environment"
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies
print_status "Installing dependencies..."
echo ""

# Install packages with progress
pip install opencv-python
print_success "OpenCV installed"

# Install MediaPipe if compatible
if [ "$USE_MEDIAPIPE" = true ]; then
    if pip install mediapipe; then
        print_success "MediaPipe installed"
    else
        print_warning "MediaPipe installation failed - will use OpenCV face detection"
        USE_MEDIAPIPE=false
    fi
else
    print_warning "Skipping MediaPipe (incompatible Python version)"
    print_status "Will use OpenCV's DNN face detection instead"
fi

# Special handling for dlib/face_recognition on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS - installing face_recognition with optimizations..."
    
    # Check for Homebrew
    if command -v brew &> /dev/null; then
        print_status "Checking for cmake..."
        if ! command -v cmake &> /dev/null; then
            print_warning "cmake not found, installing via Homebrew..."
            brew install cmake
        fi
    else
        print_warning "Homebrew not found. You may need to install cmake manually."
    fi
    
    # Install dlib with optimizations
    pip install dlib --config-settings="--build-option=--yes" --config-settings="--build-option=USE_AVX_INSTRUCTIONS"
    print_success "dlib installed with optimizations"
fi

pip install face-recognition
print_success "face_recognition installed"

pip install Pillow
print_success "Pillow installed"

echo ""
print_success "All dependencies installed successfully!"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p compute-node/known_faces
mkdir -p compute-node/models
print_success "Directories created"

# Check configuration
print_status "Checking configuration..."
if [ -f "config.py" ]; then
    # Extract PI_IP from config
    PI_IP=$(python3 -c "from config import PI_IP; print(PI_IP)" 2>/dev/null || echo "")
    if [ "$PI_IP" = "192.168.1.127" ]; then
        print_warning "Don't forget to update PI_IP in config.py with your Raspberry Pi's address!"
    else
        print_success "PI_IP configured: $PI_IP"
    fi
else
    print_error "config.py not found!"
fi

# Install face recognition models
print_status "Installing face recognition models..."
if pip install git+https://github.com/ageitgey/face_recognition_models; then
    print_success "Face recognition models installed"
else
    print_warning "Face recognition models install failed - may need manual installation"
fi

# Setup complete
echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Update config.py with your Pi's IP address"
echo "   (The Pi will stream TO this compute node)"
echo ""
echo "3. Train faces:"
echo "   python compute-node/face_teacher.py     (basic training)"
echo "   python compute-node/angle_master.py     (multi-angle training)"
echo ""
echo "4. Run recognition:"
echo "   python compute-node/face_spy.py         (faces only)"
echo "   python compute-node/eagle_eye.py        (faces + objects)"
echo ""
echo "5. On your Raspberry Pi, run:"
echo "   ./pi-streamer/pi_cam_stream.sh"
echo ""

# Final check
if [ -f "requirements.txt" ]; then
    print_success "All required files present"
else
    print_warning "requirements.txt not found - some files may be missing"
fi

# Deactivate for clean exit
deactivate 2>/dev/null || true