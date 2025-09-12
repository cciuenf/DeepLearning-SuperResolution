"""
GUI Module
Contains the ImageViewer class for visual image comparison
"""

import tkinter as tk
from tkinter import Label, Canvas, Button, Frame
from PIL import Image, ImageTk
import cv2 as cv
from metrics import calculate_psnr, calculate_ssim, calculate_all_psnr


class ImageViewer:
    """GUI class for comparing original and upscaled images"""
    
    def __init__(self, root, originals, upscaled):
        """
        Initialize the image viewer
        
        Args:
            root: Tkinter root window
            originals: List of original image dictionaries
            upscaled: List of upscaled image dictionaries
        """
        self.root = root
        self.originals = originals
        self.upscaled = upscaled
        self.index = 0
        
        if len(self.originals) != len(self.upscaled):
            print(f"Warning: Mismatched image counts - {len(self.originals)} originals, {len(self.upscaled)} upscaled")
        
        self._setup_ui()
        self._show_images()
        self._calculate_all_psnr()
    
    def _setup_ui(self):
        """Set up the user interface components"""
        self.root.title("Image Comparison Tool")
        self.root.geometry("1200x700")
        
        self.label_image = Label(self.root, text="", font=("Arial", 14))
        self.label_psnr = Label(self.root, text="", font=("Arial", 14))
        self.label_ssim = Label(self.root, text="", font=("Arial", 14))
        self.label_image.pack(pady=5)
        self.label_psnr.pack()
        self.label_ssim.pack()
        
        self.canvas = Canvas(self.root, width=1000, height=500, bg='lightgray')
        self.canvas.pack(pady=10)
        
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=5)
        
        self.btn_prev = Button(
            btn_frame, 
            text="<< Previous", 
            command=self._show_prev,
            font=("Arial", 10),
            state="normal" if len(self.originals) > 1 else "disabled"
        )
        self.btn_prev.grid(row=0, column=0, padx=5)
        
        self.btn_next = Button(
            btn_frame, 
            text="Next >>", 
            command=self._show_next,
            font=("Arial", 10),
            state="normal" if len(self.originals) > 1 else "disabled"
        )
        self.btn_next.grid(row=0, column=1, padx=5)
        
        self.root.bind('<Left>', lambda e: self._show_prev())
        self.root.bind('<Right>', lambda e: self._show_next())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.focus_set()  # Allow keyboard events
        
        self.label_all_psnr = Label(self.root, text="", font=("Arial", 12, "italic"))
        self.label_all_psnr.pack(pady=5)
        
        instructions = Label(
            self.root, 
            text="Use arrow keys or buttons to navigate • ESC to quit",
            font=("Arial", 10),
            fg="gray"
        )
        instructions.pack()
    
    def _calculate_all_psnr(self):
        """Calculate and display average PSNR across all images"""
        try:
            avg_psnr = calculate_all_psnr(self.originals, self.upscaled)
            self.label_all_psnr.config(text=f"Average PSNR across all images: {avg_psnr:.2f} dB")
        except Exception as e:
            self.label_all_psnr.config(text=f"Error calculating average PSNR: {str(e)}")
    
    def _show_images(self):
        """Display the current pair of images with their metrics"""
        if not self.originals or not self.upscaled:
            return
        
        max_index = min(len(self.originals), len(self.upscaled)) - 1
        if self.index > max_index:
            self.index = max_index
        
        try:
            orig = self.originals[self.index]["cv_img"]
            upsc = self.upscaled[self.index]["cv_img"]
            
            psnr_val = calculate_psnr(orig, upsc)
            ssim_val = calculate_ssim(orig, upsc)
            
            total_images = min(len(self.originals), len(self.upscaled))
            self.label_image.config(text=f"Image {self.index+1} of {total_images}")
            self.label_psnr.config(text=f"PSNR: {psnr_val:.2f} dB")
            
            if hasattr(ssim_val, '__len__') and len(ssim_val) > 0:
                ssim_display = ssim_val[0]
            else:
                ssim_display = ssim_val
            self.label_ssim.config(text=f"SSIM: {ssim_display:.4f}")
            
            orig_rgb = cv.cvtColor(orig, cv.COLOR_BGR2RGB)
            upsc_rgb = cv.cvtColor(upsc, cv.COLOR_BGR2RGB)
            
            display_size = 400
            orig_pil = Image.fromarray(orig_rgb)
            upsc_pil = Image.fromarray(upsc_rgb)
            
            orig_pil.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            upsc_pil.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
            
            self.tk_orig = ImageTk.PhotoImage(orig_pil)
            self.tk_upsc = ImageTk.PhotoImage(upsc_pil)
            
            self.canvas.delete("all")
            
            orig_x = 250 - orig_pil.width // 2
            orig_y = 250 - orig_pil.height // 2
            upsc_x = 750 - upsc_pil.width // 2
            upsc_y = 250 - upsc_pil.height // 2
            
            self.canvas.create_image(orig_x, orig_y, anchor=tk.NW, image=self.tk_orig)
            self.canvas.create_image(upsc_x, upsc_y, anchor=tk.NW, image=self.tk_upsc)
            
            self.canvas.create_text(250, 20, text="Original", font=("Arial", 14, "bold"), fill="black")
            self.canvas.create_text(750, 20, text="Upscaled", font=("Arial", 14, "bold"), fill="black")
            
            orig_info = f"{orig.shape[1]}×{orig.shape[0]}"
            upsc_info = f"{upsc.shape[1]}×{upsc.shape[0]}"
            self.canvas.create_text(250, 470, text=orig_info, font=("Arial", 10), fill="gray")
            self.canvas.create_text(750, 470, text=upsc_info, font=("Arial", 10), fill="gray")
            
        except Exception as e:
            print(f"Error displaying images: {e}")
            self.label_image.config(text=f"Error loading image {self.index+1}")
            self.label_psnr.config(text="PSNR: N/A")
            self.label_ssim.config(text="SSIM: N/A")
    
    def _show_prev(self):
        """Show the previous image pair"""
        if self.index > 0:
            self.index -= 1
            self._show_images()
            self._update_button_states()
    
    def _show_next(self):
        """Show the next image pair"""
        max_index = min(len(self.originals), len(self.upscaled)) - 1
        if self.index < max_index:
            self.index += 1
            self._show_images()
            self._update_button_states()
    
    def _update_button_states(self):
        """Update the state of navigation buttons"""
        max_index = min(len(self.originals), len(self.upscaled)) - 1
        
        if self.index <= 0:
            self.btn_prev.config(state="disabled")
        else:
            self.btn_prev.config(state="normal")
        
        if self.index >= max_index:
            self.btn_next.config(state="disabled")
        else:
            self.btn_next.config(state="normal")
    
    def set_image_index(self, index):
        """Set the current image index directly"""
        max_index = min(len(self.originals), len(self.upscaled)) - 1
        if 0 <= index <= max_index:
            self.index = index
            self._show_images()
            self._update_button_states()
    
    def get_current_index(self):
        """Get the current image index"""
        return self.index
    
    def get_image_count(self):
        """Get the total number of image pairs"""
        return min(len(self.originals), len(self.upscaled))
