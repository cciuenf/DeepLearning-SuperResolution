#!/usr/bin/env bash

# Simple C++ build script for main.cpp with OpenCV
set -e

# Configuration
OUTPUT_NAME="bin.out"
SOURCE_FILE="main.cpp"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if main.cpp exists
if [ ! -f "$SOURCE_FILE" ]; then
    print_error "$SOURCE_FILE not found!"
    echo "Creating a basic main.cpp with OpenCV..."
    
    cat > main.cpp << 'EOF'
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Super Resolution Functions - OpenCV Version: " << CV_VERSION << std::endl;
    
    // Create a simple test image
    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::putText(image, "Hello OpenCV 4!", 
                cv::Point(50, 200), 
                cv::FONT_HERSHEY_SIMPLEX, 
                1, 
                cv::Scalar(0, 255, 0), 
                2);
    
    // Save image (this always works)
    cv::imwrite("output.png", image);
    std::cout << "Image saved as output.png" << std::endl;
    
    // Test some basic OpenCV operations
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("output_gray.png", gray);
    std::cout << "Grayscale image saved as output_gray.png" << std::endl;
    
    // Image info
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    
    std::cout << "OpenCV is working correctly!" << std::endl;
    
    return 0;
}
EOF
    print_success "Created $SOURCE_FILE"
fi

# Build command
echo "Building $SOURCE_FILE..."

g++ -std=c++17 \
    "$(pkg-config --cflags opencv4)" \
    -o "$OUTPUT_NAME" \
    "$SOURCE_FILE" \
    "$(pkg-config --libs opencv4)"

print_success "Build completed! Executable: ./$OUTPUT_NAME"
echo "Run with: ./$OUTPUT_NAME"
