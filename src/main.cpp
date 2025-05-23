#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Super Resolution Functions - OpenCV Version: " << CV_VERSION << std::endl;

    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::putText(image, "Hello OpenCV 4!", 
                cv::Point(50, 200), 
                cv::FONT_HERSHEY_SIMPLEX, 
                1, 
                cv::Scalar(0, 255, 0), 
                2);

    cv::imwrite("output.png", image);
    std::cout << "Image saved as output.png" << std::endl;

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("output_gray.png", gray);
    std::cout << "Grayscale image saved as output_gray.png" << std::endl;

    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;

    std::cout << "OpenCV is working correctly!" << std::endl;

    return 0;
}
