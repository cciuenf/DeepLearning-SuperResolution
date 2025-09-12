"""
Image Processing Module
Handles cutting patches and creating mutations
"""

import random
import cv2 as cv
import os


class ImageProcessor:
    """Handles image cutting and mutation operations"""
    
    def __init__(self, cut_size=50, verbose=False):
        """
        Initialize the image processor
        
        Args:
            cut_size (int): Size of square patches to cut
            verbose (bool): Enable verbose output
        """
        self.cut_size = cut_size
        self.verbose = verbose
    
    def _aabb_intersect(self, box1, box2):
        """Check if two axis-aligned bounding boxes intersect"""
        return not (box1['x'] + box1['width'] <= box2['x'] or
                    box2['x'] + box2['width'] <= box1['x'] or
                    box1['y'] + box1['height'] <= box2['y'] or
                    box2['y'] + box2['height'] <= box1['y'])
    
    def _find_valid_position(self, image_shape, existing_boxes, max_attempts=1000):
        """
        Find a valid position for a new patch that doesn't intersect with existing ones
        
        Args:
            image_shape (tuple): Shape of the source image (height, width, channels)
            existing_boxes (list): List of existing bounding boxes
            max_attempts (int): Maximum number of placement attempts
            
        Returns:
            tuple: (x, y, box) or (None, None, None) if no valid position found
        """
        height, width = image_shape[:2]
        padding = int((self.cut_size - 1) / 2)
        
        for _ in range(max_attempts):
            center_x = random.randint(padding, width - padding - 1)
            center_y = random.randint(padding, height - padding - 1)
            
            x = center_x - padding
            y = center_y - padding
            
            new_box = {
                'x': x,
                'y': y,
                'width': self.cut_size,
                'height': self.cut_size
            }
            
            intersects = any(self._aabb_intersect(new_box, existing_box) for existing_box in existing_boxes)
            
            if not intersects:
                return x, y, new_box
        
        return None, None, None
    
    def cut_non_intersecting(self, image, num_cuts, output_prefix='output'):
        """
        Cut non-intersecting square patches from an image
        
        Args:
            image (dict): Image dictionary with 'cv_img' key
            num_cuts (int): Number of patches to cut
            output_prefix (str): Prefix for output filenames
            
        Returns:
            int: Number of successfully created cuts
        """
        existing_boxes = []
        successful_cuts = 0
        
        if self.verbose:
            print(f"Attempting to cut {num_cuts} patches of size {self.cut_size}x{self.cut_size}")
        
        for i in range(num_cuts):
            x, y, box = self._find_valid_position(image['cv_img'].shape, existing_boxes)
            
            if x is None:
                if self.verbose:
                    print(f"Could only place {successful_cuts} cuts out of {num_cuts}")
                break
            
            new_img = image['cv_img'][y:y+self.cut_size, x:x+self.cut_size]
            
            output_filename = f'{output_prefix}_{i}.jpg'
            cv.imwrite(output_filename, new_img)
            
            existing_boxes.append(box)
            successful_cuts += 1
            
            if self.verbose:
                print(f"Cut {i+1}: Position ({x}, {y}) - Size: {new_img.shape} -> {output_filename}")
        
        return successful_cuts
    
    def create_mutated_versions(self, original_cuts_count, output_folder="mutated", input_prefix='output'):
        """
        Create rotated and flipped versions of existing cut patches
        
        Args:
            original_cuts_count (int): Number of original cuts to process
            output_folder (str): Output directory for mutations
            input_prefix (str): Prefix of input files
            
        Returns:
            int: Total number of mutations created
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            if self.verbose:
                print(f"Created output folder: {output_folder}")
        
        total_mutations = 0
        
        for i in range(original_cuts_count):
            original_filename = f'{input_prefix}_{i}.jpg'
            
            if not os.path.exists(original_filename):
                if self.verbose:
                    print(f"Warning: {original_filename} not found, skipping mutations")
                continue
            
            original_img = cv.imread(original_filename)
            if original_img is None:
                if self.verbose:
                    print(f"Warning: Could not load {original_filename}, skipping")
                continue
            
            if self.verbose:
                print(f"Creating mutations for {original_filename}...")
            
            rotations = {
                90: cv.rotate(original_img, cv.ROTATE_90_CLOCKWISE),
                180: cv.rotate(original_img, cv.ROTATE_180),
                270: cv.rotate(original_img, cv.ROTATE_90_COUNTERCLOCKWISE)
            }
            
            for angle, rotated_img in rotations.items():
                rotated_filename = os.path.join(output_folder, f'{input_prefix}_{i}_rot_{angle}.jpg')
                cv.imwrite(rotated_filename, rotated_img)
                total_mutations += 1
                if self.verbose:
                    print(f"  Saved: {rotated_filename}")
            
            h_flipped = cv.flip(original_img, 1)
            h_flip_filename = os.path.join(output_folder, f'{input_prefix}_{i}_mirror_h.jpg')
            cv.imwrite(h_flip_filename, h_flipped)
            total_mutations += 1
            if self.verbose:
                print(f"  Saved: {h_flip_filename}")
            
            v_flipped = cv.flip(original_img, 0)
            v_flip_filename = os.path.join(output_folder, f'{input_prefix}_{i}_mirror_v.jpg')
            cv.imwrite(v_flip_filename, v_flipped)
            total_mutations += 1
            if self.verbose:
                print(f"  Saved: {v_flip_filename}")
            
            original_copy_filename = os.path.join(output_folder, f'{input_prefix}_{i}_original.jpg')
            cv.imwrite(original_copy_filename, original_img)
            total_mutations += 1
            if self.verbose:
                print(f"  Saved: {original_copy_filename}")
        
        if self.verbose:
            print(f"\nMutation complete! Created {total_mutations} total mutations")
        
        return total_mutations
    
    def set_cut_size(self, new_size):
        """Update the cut size"""
        self.cut_size = new_size
    
    def set_verbose(self, verbose):
        """Enable or disable verbose output"""
        self.verbose = verbose
