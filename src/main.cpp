#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

float MeanSquaredError(cv::Mat img_og, cv::Mat img_noise){
  float sum = 0;
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        sum += pow((img_og.at<cv::Vec3b>(i, j)[k] - img_noise.at<cv::Vec3b>(i, j)[k]), 2);
      }
    }
  }

  return sum /= 3 * img_og.cols * img_og.rows;
}

float MeanAbsoluteError(cv::Mat img_og, cv::Mat img_noise){
  float sum = 0;
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        sum += abs((img_og.at<cv::Vec3b>(i, j)[k] - img_noise.at<cv::Vec3b>(i, j)[k]));
      }
    }
  }

  return sum /= 3 * img_og.cols * img_og.rows;
}

float PSNR(cv::Mat img_og, cv::Mat img_noise){
  float sum = 0;
  float psnr = 0;
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        sum += pow((img_og.at<cv::Vec3b>(i, j)[k] - img_noise.at<cv::Vec3b>(i, j)[k]), 2);
      }
    }
  }

  psnr = 10 * (log(pow(255, 2) / sum) / log(10));
  return psnr;
}

int main() {
  std::srand(std::clock());

  std::string image_path_1 = cv::samples::findFile("images/tokyo-sniper.jpg");
  std::string image_path_2 = cv::samples::findFile("images/tokyo-sniper-noise.png");
  cv::Mat img_og = cv::imread(image_path_1, cv::IMREAD_COLOR);
  cv::Mat img_noise = cv::imread(image_path_2, cv::IMREAD_COLOR);

  float mean_squared_error, mean_absolute_error, psnr;

  if(img_og.empty() || img_noise.empty()){
      std::cout << "Could not read one of the images" << std::endl;
      return 1;
  }

  mean_squared_error = MeanSquaredError(img_og, img_noise);
  mean_absolute_error = MeanAbsoluteError(img_og, img_noise);
  psnr = PSNR(img_og, img_noise);

  std::cout << "Mean Squared Error: " << mean_squared_error << std::endl;
  std::cout << "Mean Absolute Error: " << mean_absolute_error << std::endl;
  std::cout << "Peak Signal to Noise Ratio: " << psnr << std::endl;

  return 0;
}
