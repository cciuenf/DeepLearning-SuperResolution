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
  %(prog)s --gui                          # Launch GUI for image comparison
  %(prog)s --cut-images --count 10        # Cut 10 non-intersecting patches
  %(prog)s --create-mutations             # Create mutations from existing cuts
  %(prog)s --metrics                      # Calculate and display metrics
  %(prog)s --cut-images --count 5 --rotation-angle 45 --incremental --verbose
        """
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Launch the GUI for image comparison'
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
        '--incremental',
        action='store_true',
        help='Use incremental 1-degree rotations (each builds on previous)'
    )
    
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save each 1-degree step when using --incremental'
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
                mode = "incremental" if args.incremental else "direct"
                print(f"\nApplying {mode} rotation to {successful_cuts} cuts...")
                rotator.process_cuts_with_rotation(
                    cuts_count=successful_cuts,
                    output_folder=args.rotation_folder,
                    rotation_angle=args.rotation_angle,
                    rotation_range=rotation_range_val,
                    incremental=args.incremental,
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
        originals = image_lib.get_images('original')
        upscaled = image_lib.get_images('upscaled')
        
        if not originals or not upscaled:
            print("Error: Need both 'original' and 'upscaled' images for GUI")
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
