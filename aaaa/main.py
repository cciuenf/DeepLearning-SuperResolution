import random
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

def aabb_intersect(box1, box2):
    return not (box1['x'] + box1['width'] <= box2['x'] or
                box2['x'] + box2['width'] <= box1['x'] or
                box1['y'] + box1['height'] <= box2['y'] or
                box2['y'] + box2['height'] <= box1['y'])

def find_valid_position(image_shape, cut_size, existing_boxes, max_attempts=1000):
    height, width = image_shape[:2]
    padding = int((cut_size - 1) / 2)
    
    for _ in range(max_attempts):
        center_x = random.randint(padding, width - padding - 1)
        center_y = random.randint(padding, height - padding - 1)
        
        x = center_x - padding
        y = center_y - padding
        
        new_box = {
            'x': x,
            'y': y,
            'width': cut_size,
            'height': cut_size
        }
        
        intersects = any(aabb_intersect(new_box, existing_box) for existing_box in existing_boxes)
        
        if not intersects:
            return x, y, new_box
    
    return None, None, None

cut_size = 50
image_count = 10
position_array = []

def cut_non_intersecting(image, num_cuts):
    existing_boxes = []
    successful_cuts = 0
    
    for i in range(num_cuts):
        x, y, box = find_valid_position(image['cv_img'].shape, cut_size, existing_boxes)
        
        if x is None:
            print(f"Could only place {successful_cuts} cuts out of {num_cuts}")
            break
        
        new_img = image['cv_img'][y:y+cut_size, x:x+cut_size]
        
        output_filename = f'output_{i}.jpg'
        cv.imwrite(output_filename, new_img)
        
        existing_boxes.append(box)
        successful_cuts += 1
        
        print(f"Cut {i+1}: Position ({x}, {y}) - Size: {new_img.shape}")
    
    return successful_cuts

def create_mutated_versions(original_cuts_count, output_folder="mutated"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_mutations = 0
    
    for i in range(original_cuts_count):
        original_filename = f'output_{i}.jpg'
        
        if not os.path.exists(original_filename):
            print(f"Warning: {original_filename} not found, skipping mutations")
            continue
        
        original_img = cv.imread(original_filename)
        if original_img is None:
            print(f"Warning: Could not load {original_filename}, skipping")
            continue
        
        print(f"Creating mutations for {original_filename}...")
        
        rotations = {
            90: cv.rotate(original_img, cv.ROTATE_90_CLOCKWISE),
            180: cv.rotate(original_img, cv.ROTATE_180),
            270: cv.rotate(original_img, cv.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        for angle, rotated_img in rotations.items():
            rotated_filename = os.path.join(output_folder, f'output_{i}_rot_{angle}.jpg')
            cv.imwrite(rotated_filename, rotated_img)
            total_mutations += 1
            print(f"  Saved: {rotated_filename}")
        
        h_flipped = cv.flip(original_img, 1)
        h_flip_filename = os.path.join(output_folder, f'output_{i}_mirror_h.jpg')
        cv.imwrite(h_flip_filename, h_flipped)
        total_mutations += 1
        print(f"  Saved: {h_flip_filename}")
        
        v_flipped = cv.flip(original_img, 0)
        v_flip_filename = os.path.join(output_folder, f'output_{i}_mirror_v.jpg')
        cv.imwrite(v_flip_filename, v_flipped)
        total_mutations += 1
        print(f"  Saved: {v_flip_filename}")
        
        original_copy_filename = os.path.join(output_folder, f'output_{i}_original.jpg')
        cv.imwrite(original_copy_filename, original_img)
        total_mutations += 1
        print(f"  Saved: {original_copy_filename}")
    
    print(f"\nMutation complete! Created {total_mutations} total mutations")
    return total_mutations

# Run the GUI
if __name__ == "__main__":
    if len(downies) > 0:
        successful_cuts = cut_non_intersecting(downies[0], image_count)
        print(f"Successfully created {successful_cuts} non-intersecting cuts")
        
        if successful_cuts > 0:
            print("\n" + "="*50)
            print("Creating mutated versions...")
            print("="*50)
            
            total_mutations = create_mutated_versions(successful_cuts)
            
            print(f"\nSummary:")
            print(f"- Original cuts: {successful_cuts}")
            print(f"- Total mutations: {total_mutations}")
            print(f"- Mutations per original: {total_mutations // successful_cuts if successful_cuts > 0 else 0}")
        
    else:
        print("No images found in the original directory")
    root = Tk()
    viewer = ImageViewer(root, downies, uppies)
    root.mainloop()
