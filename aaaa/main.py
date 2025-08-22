
import cv2 as cv
import sys, os
import math
import numpy as np
from sewar.full_ref import mse, psnr
from tkinter import *
from PIL import Image, ImageTk

directories = {
    "original": "./images/original",
    "downscaled(50%)": "./images/downscaled(50%)",
    "upscaled": "./images/upscaled",
    "upscaled(bicubic)": "./images/upscaled(bicubic)",
}

image_library = {}
for dir in directories:
    image_library[dir] = {
        "lib_path": directories[dir],
        "images": {}
    }

for dir in directories:
    filenames = os.listdir(directories[dir])
    sorted_filenames = sorted(filenames, key=lambda name: int(name.split('.')[0]))
    for file in sorted_filenames:
        image_library[dir]["images"][file] = {
            "path": os.path.join(image_library[dir]["lib_path"], file),
            "cv_img": cv.imread(os.path.join(image_library[dir]["lib_path"], file))
        }

def GetLibImages(lib_name):
    return list(image_library[lib_name]["images"].values())

downies = GetLibImages("original")
uppies = GetLibImages("upscaled")

def covariance(orig, upsc, mu1, mu2):
    diff1 = orig - mu1
    diff2 = upsc - mu2

    prod = np.multiply(diff1, diff2)
    return np.sum(prod) / diff1.size - 1

def ssim(orig, upsc, lightness=1.0, contrast=1.0, structure=1.0, K1=0.01, K2=0.03, L=255):
    C1 = pow(K1 * L, 2)
    C2 = pow(K2 * L, 2)
    C3 = C2 / 2.0

    (mu1a, sigma1a) = cv.meanStdDev(orig)
    (mu2a, sigma2a) = cv.meanStdDev(upsc)
    mu1 = mu1a[0]
    mu2 = mu2a[0]
    sigma1 = sigma1a[0]
    sigma2 = sigma2a[0]

    cov = covariance(orig, upsc, mu1, mu2)
    l = (2.0 * mu1 * mu2 + C1) / (pow(mu1, 2) + pow(mu2, 2) + C1)
    c = (2.0 * sigma1 * sigma2 + C2) / (pow(sigma1, 2) + pow(sigma2, 2) + C2)
    s = (cov + C3) / (sigma1 * sigma2 + C3)

    return pow(l, lightness) * pow(c, contrast) * pow(s, structure)

class ImageViewer:
    def __init__(self, root, originals, upscaled):
        self.root = root
        self.originals = originals
        self.upscaled = upscaled
        self.index = 0

        self.root.title("Image Comparison")

        self.label_image = Label(root, text="", font=("Arial", 14))
        self.label_psnr = Label(root, text="", font=("Arial", 14))
        self.label_ssim = Label(root, text="", font=("Arial", 14))
        self.label_image.pack()
        self.label_psnr.pack()
        self.label_ssim.pack()

        self.canvas = Canvas(root, width=1000, height=500)
        self.canvas.pack()

        btn_frame = Frame(root)
        btn_frame.pack()

        self.btn_prev = Button(btn_frame, text="<< Prev", command=self.show_prev)
        self.btn_prev.grid(row=0, column=0)

        self.btn_next = Button(btn_frame, text="Next >>", command=self.show_next)
        self.btn_next.grid(row=0, column=1)

        self.label_all_psnr = Label(root, text="", font=("Arial", 14))
        self.label_all_psnr.pack()

        self.show_images()
        self.all_psnr()

    def all_psnr(self):
        psnr_total = 0.0
        for i in range(len(self.originals)):
            orig = self.originals[i]["cv_img"]
            upsc = self.upscaled[i]["cv_img"]
            psnr_total = mse(orig, upsc)
        psnrr = math.log(pow(255, 2) / (psnr_total/len(self.originals)), 10) * 10
        self.label_all_psnr.config(text=f"All Images PSNR: {psnrr:.2f}")

    def show_images(self):
        orig = self.originals[self.index]["cv_img"]
        upsc = self.upscaled[self.index]["cv_img"]

        psnr_val = psnr(orig, upsc)
        ssim_val = ssim(orig, upsc)
        self.label_image.config(text=f"Image {self.index+1}/{len(self.originals)}")
        self.label_psnr.config(text=f"PSNR: {psnr_val:.2f} dB")
        self.label_ssim.config(text=f"SSIM: {ssim_val[0]:.2f}")

        # Convert images to RGB and resize for display
        orig_rgb = cv.cvtColor(orig, cv.COLOR_BGR2RGB)
        upsc_rgb = cv.cvtColor(upsc, cv.COLOR_BGR2RGB)

        orig_img = Image.fromarray(orig_rgb).resize((400, 400))
        upsc_img = Image.fromarray(upsc_rgb).resize((400, 400))

        self.tk_orig = ImageTk.PhotoImage(orig_img)
        self.tk_upsc = ImageTk.PhotoImage(upsc_img)

        self.canvas.delete("all")
        self.canvas.create_image(50, 50, anchor=NW, image=self.tk_orig)
        self.canvas.create_image(500, 50, anchor=NW, image=self.tk_upsc)

    def show_prev(self):
        if self.index > 0:
            self.index -= 1
            self.show_images()

    def show_next(self):
        if self.index < len(self.originals) - 1:
            self.index += 1
            self.show_images()

# Run the GUI
if __name__ == "__main__":
    root = Tk()
    viewer = ImageViewer(root, downies, uppies)
    root.mainloop()



"""
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
"""
