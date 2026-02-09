from PIL import Image
import os
import sys

def resize_image(input_path, output_path, width, height, maintain_aspect=True):
    """Resize a single image"""
    try:
        img = Image.open(input_path)
        
        if maintain_aspect:
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            resized_img = img
        else:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        
        resized_img.save(output_path)
        print(f"✓ Resized: {os.path.basename(input_path)} -> {resized_img.size}")
        return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def process_images(paths, width, height, output_dir=None, maintain_aspect=True):
    """Process multiple images or folders"""
    processed = 0
    
    for path in paths:
        if os.path.isfile(path):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_resized{ext}")
            else:
                name, ext = os.path.splitext(path)
                output_path = f"{name}_resized{ext}"
            
            if resize_image(path, output_path, width, height, maintain_aspect):
                processed += 1
                
        elif os.path.isdir(path):
            output_folder = output_dir if output_dir else os.path.join(path, "resized_output")
            os.makedirs(output_folder, exist_ok=True)
            
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                        output_path = os.path.join(output_folder, f"{name}_resized{ext}")
                        if resize_image(file_path, output_path, width, height, maintain_aspect):
                            processed += 1
    
    return processed

def main():
    print("=== Image Resizer ===\n")
    
    if len(sys.argv) > 3:
        width = int(sys.argv[1])
        height = int(sys.argv[2])
        paths = sys.argv[3:]
        maintain_aspect = True
        output_dir = None
    else:
        print("Enter image file paths or folder paths (comma-separated):")
        input_str = input("> ").strip()
        paths = [p.strip() for p in input_str.split(',')]
        
        print("\nEnter target width:")
        width = int(input("> ").strip())
        
        print("Enter target height:")
        height = int(input("> ").strip())
        
        print("\nMaintain aspect ratio? (y/n, default=y):")
        maintain_aspect = input("> ").strip().lower() != 'n'
        
        print("\nEnter output directory (press Enter to save alongside originals):")
        output_dir = input("> ").strip() or None
    
    total = process_images(paths, width, height, output_dir, maintain_aspect)
    print(f"\n✓ Successfully processed {total} images")

if __name__ == "__main__":
    main()
