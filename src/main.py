import cv2 as cv
import sys, os
 
directories = {
    "original": "./images/original",
    "downscaled(50%)": "./images/downscaled(50%)",
    "upscaled": "./images/upscaled",
    "upscaled(bicubic)": "./images/upscaled(bicubic)",
}

for dir in directories:
    imgs = []
    imgs.append(os.listdir(directories[dir]))
    directories[dir] = imgs

print(directories)

all_imgs = []
for dir in directories:
    images = []
    for list in directories[dir]:
        for image in list:
            images.append(cv.imread(cv.samples.findFile("./images/"+dir+"/"+image)))
    all_imgs.append(images)

# for list in all_imgs:
#     for img in list:
#         cv.imshow("", img)
#         k = cv.waitKey(0)

def ShowImg(img):
    cv.imshow("", img)
    k = cv.waitKey(0)

def MeanSquaredError(img1, img2):
    ShowImg(img1)
    ShowImg(img2)
    height, width, channels = img1.shape
    sum = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                sum += pow(float(img1[i, j, k]) - float(img2[i, j, k]), 2)
    return sum / (channels * height * width)

MSE = MeanSquaredError(all_imgs[0][0], all_imgs[2][1])
print(MSE)

# float MeanSquaredError(cv::Mat img_og, cv::Mat img_constructed){
#   CV_Assert(img_og.size() == img_constructed.size());
#   CV_Assert(img_og.type() == img_constructed.type());
#   float sum = 0;
#   float diff = 0;
#   #ifdef MIMG
#   cv::Mat diff_img(img_og.size(), img_og.type());
#   #endif
#   for(int i = 0; i < img_og.cols; i++){
#     for(int j = 0; j < img_og.rows; j++){
#       for(int k = 0; k < 3; k++){
#         diff = pow((img_constructed.at<cv::Vec3b>(i, j)[k]) - img_og.at<cv::Vec3b>(i, j)[k], 2);
#         #ifdef MIMG
#         diff_img.at<cv::Vec3b>(i, j)[k] = diff;
#         #endif
#         sum += diff;
#       }
#     }
#   }
#
#   #ifdef MIMG
#   cv::imwrite("output_mse.png", diff_img);
#   #endif
#
#   return sum /= 3 * img_og.cols * img_og.rows;
# }

