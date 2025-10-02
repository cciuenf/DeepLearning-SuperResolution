"""
Custom Rotation Processing Module
Handles arbitrary rotation angles with intelligent cropping and custom incremental rotation
"""

import cv2 as cv
import numpy as np
import os
import random
import math


class RotationProcessor:
    """Handles custom rotation operations with intelligent cropping"""

    def __init__(self, verbose=False):
        """
        Initialize the rotation processor

        Args:
            verbose (bool): Enable verbose output
        """
        self.verbose = verbose

    def _get_rotation_matrix(self, image, angle):
        """
        Get rotation matrix for given angle

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            tuple: (rotation_matrix, rotated_dimensions)
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv.getRotationMatrix2D(center, angle, 1.0)

        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        return M, (new_w, new_h)

    def _calculate_inner_rect(self, original_width, original_height, angle):
        """
        Calculate the largest rectangle that fits inside the rotated image
        without black corners

        Args:
            original_width: Original image width
            original_height: Original image height
            angle: Rotation angle in degrees

        Returns:
            tuple: (new_width, new_height) of the largest inner rectangle
        """
        angle_rad = math.radians(abs(angle))

        if angle_rad == 0:
            return original_width, original_height

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        if original_width <= original_height:
            new_width = int(original_width * cos_a + original_height * sin_a)
            new_height = int(original_height * cos_a + original_width * sin_a)
        else:
            new_width = int(original_width * cos_a + original_height * sin_a)
            new_height = int(original_height * cos_a + original_width * sin_a)

        w, h = original_width, original_height
        new_w = int((w * cos_a - h * sin_a) / (cos_a * cos_a - sin_a * sin_a))
        new_h = int((h * cos_a - w * sin_a) / (cos_a * cos_a - sin_a * sin_a))

        new_w = min(new_w, int(w * cos_a))
        new_h = min(new_h, int(h * cos_a))

        new_w = max(1, min(new_w, original_width))
        new_h = max(1, min(new_h, original_height))

        return new_w, new_h

    def _create_rotation_mask(self, image_shape, angle):
        """
        Create a mask to identify valid (non-black) regions after rotation

        Args:
            image_shape: Shape of the original image
            angle: Rotation angle in degrees

        Returns:
            numpy.ndarray: Binary mask
        """
        h, w = image_shape[:2]

        mask = np.ones((h, w), dtype=np.uint8) * 255

        M, (new_w, new_h) = self._get_rotation_matrix(mask, angle)
        rotated_mask = cv.warpAffine(mask, M, (new_w, new_h))

        return rotated_mask

    def _extract_largest_valid_square(self, rotated_image, rotated_mask):
        """
        Extract the largest square region without black corners

        Args:
            rotated_image: Rotated image with potential black corners
            rotated_mask: Mask indicating valid regions

        Returns:
            numpy.ndarray: Cropped square image
        """
        contours, _ = cv.findContours(rotated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            h, w = rotated_image.shape[:2]
            size = min(h, w) // 2
            center_y, center_x = h // 2, w // 2
            start_y = center_y - size // 2
            start_x = center_x - size // 2
            return rotated_image[start_y:start_y + size, start_x:start_x + size]

        largest_contour = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(largest_contour)

        size = min(w, h)
        center_x = x + w // 2
        center_y = y + h // 2

        start_x = center_x - size // 2
        start_y = center_y - size // 2

        start_x = max(0, min(start_x, rotated_image.shape[1] - size))
        start_y = max(0, min(start_y, rotated_image.shape[0] - size))

        return rotated_image[start_y:start_y + size, start_x:start_x + size]

    def rotate_with_crop(self, image, angle):
        """
        Rotate image by given angle and extract the largest valid square

        Args:
            image: Input image (numpy array)
            angle: Rotation angle in degrees

        Returns:
            numpy.ndarray: Rotated and cropped image
        """
        if angle == 0:
            return image

        M, (new_w, new_h) = self._get_rotation_matrix(image, angle)

        rotated = cv.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))

        mask = self._create_rotation_mask(image.shape, angle)

        cropped = self._extract_largest_valid_square(rotated, mask)

        return cropped

    # ===== CUSTOM INCREMENTAL ROTATION METHODS =====

    def _rotate_single_degree_custom(self, img, clockwise=True):
        """
        Rotate image by exactly 1 degree using custom rotation matrix

        Args:
            img: Input image (numpy array)
            clockwise: True for +1 degree, False for -1 degree

        Returns:
            Rotated image
        """
        h, w = img.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        # 1 degree in radians
        angle_rad = math.radians(1.0 if clockwise else -1.0)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Translate to origin
        x_centered = x_grid - center_x
        y_centered = y_grid - center_y

        # Apply inverse rotation matrix to find source coordinates
        src_x = x_centered * cos_a + y_centered * sin_a + center_x
        src_y = -x_centered * sin_a + y_centered * cos_a + center_y

        # Convert to float32 for cv.remap
        src_x = src_x.astype(np.float32)
        src_y = src_y.astype(np.float32)

        # Use remap for interpolation (we computed rotation ourselves)
        output = cv.remap(img, src_x, src_y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        return output

    def rotate_incrementally(self, img, target_angle, save_each=True, output_prefix="incremental", output_folder=None):
        """
        Rotate image incrementally by 1 degree steps, each building on the previous

        Args:
            img: Input image
            target_angle: Target rotation angle in degrees (can be negative)
            save_each: Whether to save each intermediate step
            output_prefix: Prefix for saved files
            output_folder: Folder to save intermediate steps

        Returns:
            Final rotated image
        """
        if target_angle == 0:
            return img

        # Determine rotation direction and steps
        steps = int(abs(target_angle))
        clockwise = target_angle > 0

        if save_each and output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            if self.verbose:
                print(f"  Created intermediate folder: {output_folder}")

        current_img = img.copy()

        if self.verbose:
            print(f"  Performing {steps} incremental 1° rotations...")

        for step in range(steps):
            # Rotate by 1 degree based on previous rotation
            current_img = self._rotate_single_degree_custom(current_img, clockwise)

            current_angle = (step + 1) * (1 if clockwise else -1)

            if save_each and output_folder:
                filename = os.path.join(output_folder, f"{output_prefix}_step_{current_angle:+04d}.jpg")
                cv.imwrite(filename, current_img)
                if self.verbose and (step + 1) % 5 == 0:  # Report every 5 steps
                    print(f"    Progress: {step + 1}/{steps} degrees")

        return current_img

    def process_cuts_with_rotation(self, input_prefix='output', output_folder='rotated_cuts', 
                                   rotation_angle=None, rotation_range=None, cuts_count=None,
                                   incremental=False, save_intermediates=False):
        """
        Process existing cuts with custom rotation

        Args:
            input_prefix: Prefix of input cut files
            output_folder: Output directory for rotated cuts
            rotation_angle: Fixed rotation angle in degrees (XOR with rotation_range)
            rotation_range: Tuple of (min_angle, max_angle) for random rotation
            cuts_count: Number of cuts to process (auto-detect if None)
            incremental: Use incremental 1-degree rotations
            save_intermediates: Save each 1-degree step (only if incremental=True)

        Returns:
            int: Number of successfully processed cuts
        """
        if rotation_angle is not None and rotation_range is not None:
            raise ValueError("Cannot specify both rotation_angle and rotation_range")

        if rotation_angle is None and rotation_range is None:
            raise ValueError("Must specify either rotation_angle or rotation_range")

        if cuts_count is None:
            cuts_count = 0
            while os.path.exists(f'{input_prefix}_{cuts_count}.jpg'):
                cuts_count += 1

        if cuts_count == 0:
            if self.verbose:
                print("No cut files found to process")
            return 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            if self.verbose:
                print(f"Created output folder: {output_folder}")

        processed_count = 0

        for i in range(cuts_count):
            input_filename = f'{input_prefix}_{i}.jpg'

            if not os.path.exists(input_filename):
                if self.verbose:
                    print(f"Warning: {input_filename} not found, skipping")
                continue

            image = cv.imread(input_filename)
            if image is None:
                if self.verbose:
                    print(f"Warning: Could not load {input_filename}, skipping")
                continue

            if rotation_angle is not None:
                angle = rotation_angle
                angle_suffix = f"rot_{angle}"
            else:
                min_angle, max_angle = rotation_range
                angle = random.uniform(min_angle, max_angle)
                angle_suffix = f"rot_{angle:.1f}"

            if self.verbose:
                mode = "incremental" if incremental else "direct"
                print(f"Processing {input_filename} with {mode} rotation: {angle:.1f}°")

            try:
                if incremental:
                    # Create subfolder for intermediates if requested
                    intermediate_folder = None
                    if save_intermediates:
                        intermediate_folder = os.path.join(output_folder, f"steps_{i}")

                    rotated = self.rotate_incrementally(
                        image, 
                        angle, 
                        save_each=save_intermediates,
                        output_prefix=f"cut_{i}",
                        output_folder=intermediate_folder
                    )
                else:
                    # Use standard rotation with crop
                    rotated = self.rotate_with_crop(image, angle)

                output_filename = os.path.join(
                    output_folder, 
                    f'{input_prefix}_{i}_{angle_suffix}.jpg'
                )

                cv.imwrite(output_filename, rotated)
                processed_count += 1

                if self.verbose:
                    original_size = f"{image.shape[1]}x{image.shape[0]}"
                    final_size = f"{rotated.shape[1]}x{rotated.shape[0]}"
                    print(f"  Saved: {output_filename} ({original_size} -> {final_size})")

            except Exception as e:
                if self.verbose:
                    print(f"  Error processing {input_filename}: {str(e)}")
                continue

        if self.verbose:
            print(f"\nRotation processing complete! Processed {processed_count} out of {cuts_count} cuts")

        return processed_count

    def process_single_image(self, image_path, angle, output_path, incremental=False):
        """
        Process a single image with rotation

        Args:
            image_path: Path to input image
            angle: Rotation angle in degrees
            output_path: Path for output image
            incremental: Use incremental rotation

        Returns:
            bool: Success status
        """
        try:
            image = cv.imread(image_path)
            if image is None:
                return False

            if incremental:
                rotated = self.rotate_incrementally(image, angle)
            else:
                rotated = self.rotate_with_crop(image, angle)

            cv.imwrite(output_path, rotated)

            if self.verbose:
                print(f"Processed {image_path} -> {output_path} (angle: {angle}°)")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error processing {image_path}: {str(e)}")
            return False

    def batch_process_with_angle_range(self, input_files, output_folder, min_angle, max_angle, 
                                       samples_per_image=5, incremental=False):
        """
        Process multiple images with random rotations within a range

        Args:
            input_files: List of input file paths
            output_folder: Output directory
            min_angle: Minimum rotation angle
            max_angle: Maximum rotation angle
            samples_per_image: Number of rotated versions per input image
            incremental: Use incremental rotation

        Returns:
            int: Total number of processed images
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        total_processed = 0

        for file_path in input_files:
            if not os.path.exists(file_path):
                continue

            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            for sample in range(samples_per_image):
                angle = random.uniform(min_angle, max_angle)
                output_filename = f"{name_without_ext}_rot_{angle:.1f}_sample_{sample}.jpg"
                output_path = os.path.join(output_folder, output_filename)

                if self.process_single_image(file_path, angle, output_path, incremental):
                    total_processed += 1

        return total_processed

    def set_verbose(self, verbose):
        """Enable or disable verbose output"""
        self.verbose = verbose


def validate_rotation_args(rotation_angle, rotation_range):
    """
    Validate rotation arguments

    Args:
        rotation_angle: Single rotation angle or None
        rotation_range: Tuple of (min, max) angles or None

    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    if rotation_angle is not None and rotation_range is not None:
        raise ValueError("Cannot specify both --rotation-angle and --rotation-range")

    if rotation_angle is None and rotation_range is None:
        return False

    if rotation_angle is not None:
        if not isinstance(rotation_angle, (int, float)):
            raise ValueError("rotation_angle must be a number")
        if not (-360 <= rotation_angle <= 360):
            raise ValueError("rotation_angle must be between -360 and 360 degrees")

    if rotation_range is not None:
        if not isinstance(rotation_range, (list, tuple)) or len(rotation_range) != 2:
            raise ValueError("rotation_range must be a tuple/list of two numbers")
        min_angle, max_angle = rotation_range
        if not isinstance(min_angle, (int, float)) or not isinstance(max_angle, (int, float)):
            raise ValueError("rotation_range values must be numbers")
        if min_angle >= max_angle:
            raise ValueError("rotation_range minimum must be less than maximum")
        if not (-360 <= min_angle <= 360) or not (-360 <= max_angle <= 360):
            raise ValueError("rotation_range values must be between -360 and 360 degrees")

    return True


def parse_rotation_range(range_str):
    """
    Parse rotation range string like "10,45" into tuple

    Args:
        range_str: String like "10,45" or "10:45"

    Returns:
        tuple: (min_angle, max_angle)
    """
    try:
        if ',' in range_str:
            parts = range_str.split(',')
        elif ':' in range_str:
            parts = range_str.split(':')
        else:
            raise ValueError("Range must contain ',' or ':' separator")

        if len(parts) != 2:
            raise ValueError("Range must contain exactly two values")

        min_angle = float(parts[0].strip())
        max_angle = float(parts[1].strip())

        return (min_angle, max_angle)

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid rotation range format: {range_str}. Use format like '10,45' or '-30:30'")
