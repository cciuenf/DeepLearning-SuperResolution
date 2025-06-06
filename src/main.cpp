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
  #ifdef MIMG
  cv::Mat diff_img(img_og.size(), img_og.type());
  #endif
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        diff = pow((img_constructed.at<cv::Vec3b>(i, j)[k]) - img_og.at<cv::Vec3b>(i, j)[k], 2);
        #ifdef MIMG
        diff_img.at<cv::Vec3b>(i, j)[k] = diff;
        #endif
        sum += diff;
      }
    }
  }

  #ifdef MIMG
  cv::imwrite("output_mse.png", diff_img);
  #endif

  return sum /= 3 * img_og.cols * img_og.rows;
}

float MeanAbsoluteError(cv::Mat img_og, cv::Mat img_constructed){
  CV_Assert(img_og.size() == img_constructed.size());
  CV_Assert(img_og.type() == img_constructed.type());
  float sum = 0;
  float diff = 0;
  #ifdef MIMG
  cv::Mat diff_img(img_og.size(), img_og.type());
  #endif
  for(int i = 0; i < img_og.cols; i++){
    for(int j = 0; j < img_og.rows; j++){
      for(int k = 0; k < 3; k++){
        diff = abs((img_constructed.at<cv::Vec3b>(i, j)[k] - img_og.at<cv::Vec3b>(i, j)[k]));
        #ifdef MIMG
        diff_img.at<cv::Vec3b>(i, j)[k] = diff;
        #endif
        sum += diff;
      }
    }
  }

  #ifdef MIMG
  cv::imwrite("output_mae.png", diff_img);
  #endif

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

double covariance(const cv::Mat& img1, const cv::Mat& img2, double mu1, double mu2) {
    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);

    cv::Mat diff1 = img1f - mu1;
    cv::Mat diff2 = img2f - mu2;

    cv::Mat prod = diff1.mul(diff2);
    return static_cast<double>(cv::sum(prod)[0]) / (img1.total() - 1);
}

double SSIM(const cv::Mat& img1, const cv::Mat& img2,
            double alpha = 1.0, double beta = 1.0, double gamma = 1.0,
            double k1 = 0.01, double k2 = 0.03, double L = 255.0) {
    CV_Assert(img1.size() == img2.size());
    CV_Assert(img1.type() == img2.type());

    double C1 = std::pow(k1 * L, 2);
    double C2 = std::pow(k2 * L, 2);
    double C3 = C2 / 2.0;

    cv::Scalar mu1_s, sigma1_s, mu2_s, sigma2_s;
    cv::meanStdDev(img1, mu1_s, sigma1_s);
    cv::meanStdDev(img2, mu2_s, sigma2_s);

    double mu1 = mu1_s[0];
    double mu2 = mu2_s[0];
    double sigma1 = sigma1_s[0];
    double sigma2 = sigma2_s[0];

    double cov = covariance(img1, img2, mu1, mu2);

    double luminance = (2.0 * mu1 * mu2 + C1) / (std::pow(mu1, 2) + std::pow(mu2, 2) + C1);
    double contrast = (2.0 * sigma1 * sigma2 + C2) / (std::pow(sigma1, 2) + std::pow(sigma2, 2) + C2);
    double structure = (cov + C3) / (sigma1 * sigma2 + C3);

    return std::pow(luminance, alpha) * std::pow(contrast, beta) * std::pow(structure, gamma);
}

int main() {
  std::srand(std::clock());

  std::string image_path_1 = cv::samples::findFile("images/tokyo-sniper.jpg");
  std::string image_path_2 = cv::samples::findFile("images/tokyo-sniper-upscaled.jpg");
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
  double ssim_val = SSIM(img_og, img_noise, 1.0, 1.0, 1.0); // α, β, γ
  std::cout << "SSIM: " << ssim_val << std::endl;

  return 0;
}
