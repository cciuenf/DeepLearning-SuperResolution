import cv2 as cv
import sys, os
import math
import numpy as np
from sewar.full_ref import msssim, ssim, mse, vifp, psnr
 
directories = {
    "original": "./images/original",
    "downscaled(50%)": "./images/downscaled(50%)",
    "upscaled": "./images/upscaled",
    "upscaled(bicubic)": "./images/upscaled(bicubic)",
}

# image_library = {
#     example_lib = {
#         lib_path = "",
#         images = {
#             filename = {
#                 path = "",
#                 cv_img = [][][]
#             },
#         },
#     },
# }

image_library = {}
for dir in directories:
    image_library[dir] = {}
    image_library[dir]["lib_path"] = directories[dir]
    image_library[dir]["images"] = {}

for dir in directories:
    filenames = os.listdir(directories[dir])
    sorted_filenames = sorted(filenames, key=lambda name: int(name.split('.')[0]))
    for file in sorted_filenames:
        image_library[dir]["images"][file] = {}
        image_library[dir]["images"][file]["path"] = image_library[dir]["lib_path"]+"/"+file
        filepath = image_library[dir]["images"][file]["path"]
        image_library[dir]["images"][file]["cv_img"] = cv.imread(cv.samples.findFile(filepath))

def ShowImg(img):
    cv.imshow("", img)
    k = cv.waitKey(0)

def GetLibImages(lib_name):
    images = []
    for ilm, img in image_library[lib_name]["images"].items():
        images.append(img)
    return images

downies = GetLibImages("original")
uppies = GetLibImages("upscaled")
for i in range(len(downies)):
    res = psnr(downies[i]["cv_img"], uppies[i]["cv_img"])
    print(res)
