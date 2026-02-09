from PIL import Image, ImageFilter
import os
import sys

def apply_gaussian_blur(input_path, output_path, radius=2):
    """Apply Gaussian blur to a single image"""
    try:
        img = Image.open(input_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_img.save(output_path)
        print(f"✓ Blurred: {os.path.basename(input_path)} (radius={radius})")
        return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def process_images(paths, radius, output_dir=None):
    """Process multiple images or folders"""
    processed = 0
    
    for path in paths:
        if os.path.isfile(path):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_blurred{ext}")
            else:
                name, ext = os.path.splitext(path)
                output_path = f"{name}_blurred{ext}"
            
            if apply_gaussian_blur(path, output_path, radius):
                processed += 1
                
        elif os.path.isdir(path):
            output_folder = output_dir if output_dir else os.path.join(path, "blurred_output")
            os.makedirs(output_folder, exist_ok=True)
            
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                        output_path = os.path.join(output_folder, f"{name}_blurred{ext}")
                        if apply_gaussian_blur(file_path, output_path, radius):
                            processed += 1
    
    return processed

def main():
    print("=== Gaussian Blur Image Processor ===\n")
    
    if len(sys.argv) > 2:
        radius = float(sys.argv[1])
        paths = sys.argv[2:]
        output_dir = None
    else:
        print("Enter image file paths or folder paths (comma-separated):")
        input_str = input("> ").strip()
        paths = [p.strip() for p in input_str.split(',')]
        
        print("\nEnter blur radius (default=2, higher=more blur):")
        radius_str = input("> ").strip()
        radius = float(radius_str) if radius_str else 2.0
        
        print("\nEnter output directory (press Enter to save alongside originals):")
        output_dir = input("> ").strip() or None
    
    total = process_images(paths, radius, output_dir)
    print(f"\n✓ Successfully processed {total} images")

if __name__ == "__main__":
    main()
