#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

float MeanSquaredError(cv::Mat img_og, cv::Mat img_constructed){
  CV_Assert(img_og.size() == img_constructed.size());
  CV_Assert(img_og.type() == img_constructed.type());
  float sum = 0;
  float diff = 0;
  cv::Mat diff_img(img_og.size(), img_og.type());
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        diff = pow((img_constructed.at<cv::Vec3b>(i, j)[k]) - img_og.at<cv::Vec3b>(i, j)[k], 2);
        diff_img.at<cv::Vec3b>(i, j)[k] = diff;
        sum += diff;
      }
    }
  }

  cv::imwrite("output_mse.png", diff_img);

  return sum /= 3 * img_og.cols * img_og.rows;
}

float MeanAbsoluteError(cv::Mat img_og, cv::Mat img_constructed){
  CV_Assert(img_og.size() == img_constructed.size());
  CV_Assert(img_og.type() == img_constructed.type());
  float sum = 0;
  float diff = 0;
  cv::Mat diff_img(img_og.size(), img_og.type());
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        diff = abs((img_constructed.at<cv::Vec3b>(i, j)[k] - img_og.at<cv::Vec3b>(i, j)[k]));
        diff_img.at<cv::Vec3b>(i, j)[k] = diff;
        sum += diff;
      }
    }
  }

  cv::imwrite("output_mae.png", diff_img);

  return sum /= 3 * img_og.cols * img_og.rows;
}

float PSNR(cv::Mat img_og, cv::Mat img_constructed){
  CV_Assert(img_og.size() == img_constructed.size());
  CV_Assert(img_og.type() == img_constructed.type());
  float sum = 0;
  float psnr = 0;
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        sum += pow((img_constructed.at<cv::Vec3b>(i, j)[k] - img_og.at<cv::Vec3b>(i, j)[k]), 2);
      }
    }
  }

  psnr = 10 * (log(pow(255, 2) / sum) / log(10));
  return psnr;
}

double covariance(cv::Mat img_og, cv::Mat img_constructed, double M_OG_I, double M_RC_I) {
  cv::Mat diff1 = img_og - M_OG_I;
  cv::Mat diff2 = img_og - M_RC_I;

  cv::Mat prod = diff1.mul(diff2);

  return (cv::sum(prod)[0]) / (img_og.total() - 1);
  
}

double SSIM(const cv::Mat img_og, const cv::Mat img_constructed) {
  CV_Assert(img_og.size() == img_constructed.size());
  CV_Assert(img_og.type() == img_constructed.type());

  double k1 = 1.0f, k2 = 1.0f, L = 255.0f;

  double C1 = std::pow(k1 * L, 2), C2 = std::pow(k2 * L, 2);

  cv::Scalar OG_I, RC_I, OG_D, RC_D;
  cv::meanStdDev(img_og, OG_I, OG_D);
  cv::meanStdDev(img_og, RC_I, RC_D);

  double CL = 0.0f, CC = 0.0f;

  CL = (2*OG_I[0]*RC_I[0] + C1)/(OG_I[0] + RC_I[0] + C1);
  CC = (2*OG_D[0]*RC_D[0] + C2)/(OG_D[0] + RC_D[0] + C2);
  return 0.0f;
}

int main() {
  std::srand(std::clock());

  std::string image_path_1 = cv::samples::findFile("images/apple-og.jpg");
  std::string image_path_2 = cv::samples::findFile("images/apple-og-upscaled.jpg");
  cv::Mat img_og = cv::imread(image_path_1, cv::IMREAD_COLOR);
  cv::Mat img_noise = cv::imread(image_path_2, cv::IMREAD_COLOR);

  float mean_squared_error, mean_absolute_error, psnr;

  if(img_og.empty() || img_noise.empty()){
      std::cout << "Could not read one of the images" << std::endl;
      return 1;
  }

  mean_squared_error = MeanSquaredError(img_og, img_noise);
  std::cout << "Mean Squared Error: " << mean_squared_error << std::endl;
  mean_absolute_error = MeanAbsoluteError(img_og, img_noise);
  std::cout << "Mean Absolute Error: " << mean_absolute_error << std::endl;
  psnr = PSNR(img_og, img_noise);
  std::cout << "Peak Signal to Noise Ratio: " << psnr << std::endl;

  return 0;
}
