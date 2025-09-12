#!/usr/bin/env python3
"""
Quick test to verify argument parsing works
"""

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test rotation arguments")
    
    parser.add_argument('--cut-images', action='store_true', help='Cut images')
    parser.add_argument('--count', type=int, default=10, help='Count')
    parser.add_argument('--rotation-angle', type=float, help='Rotation angle')
    parser.add_argument('--rotation-range', type=str, help='Rotation range')
    parser.add_argument('--rotation-folder', type=str, default='rotated_cuts', help='Rotation folder')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print("Arguments parsed successfully:")
    print(f"  cut_images: {args.cut_images}")
    print(f"  count: {args.count}")
    print(f"  rotation_angle: {args.rotation_angle}")
    print(f"  rotation_range: {args.rotation_range}")
    print(f"  rotation_folder: {args.rotation_folder}")
    print(f"  verbose: {args.verbose}")
