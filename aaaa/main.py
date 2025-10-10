#!/usr/bin/env python3
"""
Image Processing CLI Tool
Main entry point for the application
"""

import argparse
import sys
import os
from pathlib import Path

from image_library import ImageLibrary
from image_processor import ImageProcessor
from metrics import calculate_all_psnr, calculate_metrics_summary, format_metrics_output
from gui import ImageViewer
from tkinter import Tk
from rotation_processor import RotationProcessor, parse_rotation_range, validate_rotation_args


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Image Processing and Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gui                          # Launch GUI for image comparison (original vs upscaled)
  %(prog)s --gui --compare rotated_hybrid rotated_standard  # Compare two custom directories
  %(prog)s --cut-images --count 10        # Cut 10 non-intersecting patches
  %(prog)s --create-mutations             # Create mutations from existing cuts
  %(prog)s --metrics                      # Calculate and display metrics
  %(prog)s --cut-images --count 5 --rotation-angle 52 --verbose  # Hybrid: 45° + 7×1°
  %(prog)s --cut-images --count 5 --rotation-angle 52 --rotation-mode standard  # Direct 52°
  %(prog)s --cut-images --count 5 --rotation-angle 52 --rotation-mode incremental  # 52×1°
  %(prog)s --cut-images --count 5 --rotation-angle 52 --rotation-mode all  # All three modes
        """
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Launch the GUI for image comparison'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        nargs=2,
        metavar=('DIR1', 'DIR2'),
        help='Compare two specific directories in GUI (e.g., --compare rotated_hybrid rotated_standard)'
    )
    
    parser.add_argument(
        '--cut-images',
        action='store_true',
        help='Cut non-intersecting patches from the first original image'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='Number of patches to cut (default: 10)'
    )
    
    parser.add_argument(
        '--cut-size',
        type=int,
        default=50,
        help='Size of square patches to cut (default: 50)'
    )
    
    parser.add_argument(
        '--create-mutations',
        action='store_true',
        help='Create rotated and flipped versions of existing cut patches'
    )
    
    parser.add_argument(
        '--mutations-folder',
        type=str,
        default='mutated',
        help='Output folder for mutations (default: mutated)'
    )
    
    parser.add_argument(
        '--rotation-angle',
        type=float,
        help='Fixed rotation angle in degrees (XOR with --rotation-range)'
    )
    
    parser.add_argument(
        '--rotation-range',
        type=str,
        help='Rotation range as "min,max" degrees (e.g., "10,45" or "-30:30")'
    )
    
    parser.add_argument(
        '--rotation-folder',
        type=str,
        default='rotated_cuts',
        help='Output folder for custom rotations (default: rotated_cuts)'
    )
    
    parser.add_argument(
        '--rotation-mode',
        type=str,
        choices=['hybrid', 'standard', 'incremental', 'all'],
        default='hybrid',
        help='Rotation mode: hybrid (45° + incremental, default), standard (direct rotation), incremental (all 1° steps), all (generate all three modes)'
    )
    
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save each 1-degree step when using incremental or hybrid modes'
    )
    
    parser.add_argument(
        '--metrics',
        action='store_true',
        help='Calculate and display PSNR/SSIM metrics'
    )
    
    parser.add_argument(
        '--directories',
        type=str,
        nargs='+',
        default=['original', 'downscaled(50%)', 'upscaled', 'upscaled(bicubic)'],
        help='Image directories to process (default: original downscaled(50%%) upscaled upscaled(bicubic))'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Build directories dict - handle --compare specially
    if args.compare:
        # For --compare, check multiple locations for directories
        dir1_name, dir2_name = args.compare
        directories = {}
        
        for dir_name in [dir1_name, dir2_name]:
            # Check if path exists as-is (current directory or absolute path)
            if os.path.exists(dir_name):
                directories[dir_name] = dir_name
            # Check if it exists under ./images/
            elif os.path.exists(f"./images/{dir_name}"):
                directories[dir_name] = f"./images/{dir_name}"
            # Check without leading ./
            elif os.path.exists(f"./{dir_name}"):
                directories[dir_name] = f"./{dir_name}"
            else:
                # Add it anyway, error will be caught later with helpful message
                directories[dir_name] = dir_name
    else:
        # Default behavior: use ./images/ subdirectory
        directories = {
            name: f"./images/{name}" for name in args.directories
        }
    
    image_lib = ImageLibrary(directories)
    
    if args.verbose:
        print("Image directories loaded:")
        for name, path in directories.items():
            count = len(image_lib.get_images(name))
            print(f"  {name}: {count} images from {path}")
    
    processor = ImageProcessor(cut_size=args.cut_size, verbose=args.verbose)
    rotator = RotationProcessor(verbose=args.verbose)
    
    operations_performed = False
    
    if args.cut_images:
        operations_performed = True
        originals = image_lib.get_images('original')
        if not originals:
            print("Error: No original images found")
            sys.exit(1)
        
        print(f"Cutting {args.count} non-intersecting patches from the first original image...")
        successful_cuts = processor.cut_non_intersecting(originals[0], args.count, 'output')
        print(f"Successfully created {successful_cuts} non-intersecting cuts")
        
        if args.create_mutations:
            print(f"\nCreating mutations in '{args.mutations_folder}' folder...")
            total_mutations = processor.create_mutated_versions(
                successful_cuts, 
                args.mutations_folder
            )
            print(f"Created {total_mutations} total mutations")

        try:
            rotation_range_val = parse_rotation_range(args.rotation_range) if args.rotation_range else None
            should_rotate = validate_rotation_args(args.rotation_angle, rotation_range_val)
            
            if should_rotate and successful_cuts > 0:
                print(f"\nApplying {args.rotation_mode} rotation to {successful_cuts} cuts...")
                rotator.process_cuts_with_rotation(
                    cuts_count=successful_cuts,
                    output_folder=args.rotation_folder,
                    rotation_angle=args.rotation_angle,
                    rotation_range=rotation_range_val,
                    mode=args.rotation_mode,
                    save_intermediates=args.save_intermediates
                )
        except ValueError as e:
            print(f"Error with rotation arguments: {e}")
            sys.exit(1)
            
    elif args.create_mutations:
        operations_performed = True
        print(f"Creating mutations in '{args.mutations_folder}' folder...")
        
        existing_cuts = 0
        while os.path.exists(f'output_{existing_cuts}.jpg'):
            existing_cuts += 1
        
        if existing_cuts == 0:
            print("Error: No cut images found. Run with --cut-images first.")
            sys.exit(1)
        
        total_mutations = processor.create_mutated_versions(
            existing_cuts, 
            args.mutations_folder
        )
        print(f"Created {total_mutations} total mutations from {existing_cuts} original cuts")
    
    if args.metrics:
        operations_performed = True
        originals = image_lib.get_images('original')
        upscaled = image_lib.get_images('upscaled')
        
        if not originals or not upscaled:
            print("Error: Need both 'original' and 'upscaled' images for metrics calculation")
        else:
            print("Calculating metrics...")
            summary = calculate_metrics_summary(originals, upscaled)
            print(format_metrics_output(summary, verbose=args.verbose))
    
    if args.gui:
        operations_performed = True
        
        # Determine which directories to compare
        if args.compare:
            dir1_name, dir2_name = args.compare
            
            # Get images using the directory names as keys
            dir1_images = image_lib.get_images(dir1_name)
            dir2_images = image_lib.get_images(dir2_name)
            
            if not dir1_images or not dir2_images:
                print(f"Error: Need images in both directories for comparison")
                print(f"  '{dir1_name}': {len(dir1_images)} images found")
                print(f"  '{dir2_name}': {len(dir2_images)} images found")
                
                # Show available directories
                available = [name for name in image_lib.get_library_names() if image_lib.get_image_count(name) > 0]
                if available:
                    print(f"\nAvailable directories with images:")
                    for avail_dir in available:
                        print(f"  - {avail_dir}: {image_lib.get_image_count(avail_dir)} images")
                sys.exit(1)
            
            # Check if images need resizing
            import cv2 as cv
            needs_resize = False
            target_size = None
            
            if dir1_images and dir2_images:
                img1_shape = dir1_images[0]["cv_img"].shape[:2]  # (H, W)
                img2_shape = dir2_images[0]["cv_img"].shape[:2]
                
                if img1_shape != img2_shape:
                    needs_resize = True
                    target_size = img1_shape  # Use dir1 size as reference
                    print(f"\nImage size mismatch detected:")
                    print(f"  '{dir1_name}': {img1_shape[1]}x{img1_shape[0]}")
                    print(f"  '{dir2_name}': {img2_shape[1]}x{img2_shape[0]}")
                    print(f"  Resizing '{dir2_name}' images to {img1_shape[1]}x{img1_shape[0]}...")
                    
                    # Resize all dir2 images to match dir1
                    resized_dir2_images = []
                    for img_dict in dir2_images:
                        img = img_dict["cv_img"]
                        resized_img = cv.resize(img, (img1_shape[1], img1_shape[0]), interpolation=cv.INTER_AREA)
                        resized_dir2_images.append({
                            "path": img_dict["path"],
                            "cv_img": resized_img
                        })
                    dir2_images = resized_dir2_images
                    print(f"  ✓ Resized {len(dir2_images)} images")
            
            print(f"\nLaunching GUI to compare:")
            print(f"  Left:  '{dir1_name}' ({len(dir1_images)} images)")
            print(f"  Right: '{dir2_name}' ({len(dir2_images)} images)")
            
            root = Tk()
            root.title(f"Image Comparison: {os.path.basename(dir1_name)} vs {os.path.basename(dir2_name)}")
            viewer = ImageViewer(root, dir1_images, dir2_images, label_left=dir1_name, label_right=dir2_name)
            root.mainloop()
        else:
            # Default behavior: compare 'original' and 'upscaled'
            originals = image_lib.get_images('original')
            upscaled = image_lib.get_images('upscaled')
            
            if not originals or not upscaled:
                print("Error: Need both 'original' and 'upscaled' images for GUI")
                print("Tip: Use --compare DIR1 DIR2 to compare custom directories")
                sys.exit(1)
            
            print("Launching GUI...")
            root = Tk()
            viewer = ImageViewer(root, originals, upscaled)
            root.mainloop()
    
    if not operations_performed:
        print("No operations specified. Use --help for available options.")
        print("Quick start: python main.py --gui")


if __name__ == "__main__":
    main()
