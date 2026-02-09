from PIL import Image
import os
import sys

def convert_to_grayscale(input_path, output_path):
    """Convert a single image to grayscale"""
    try:
        img = Image.open(input_path)
        grayscale_img = img.convert('L')
        grayscale_img.save(output_path)
        print(f"✓ Converted: {os.path.basename(input_path)}")
        return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def process_images(paths, output_dir=None):
    """Process multiple images or folders"""
    processed = 0
    
    for path in paths:
        if os.path.isfile(path):
            # Single file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_gray{ext}")
            else:
                name, ext = os.path.splitext(path)
                output_path = f"{name}_gray{ext}"
            
            if convert_to_grayscale(path, output_path):
                processed += 1
                
        elif os.path.isdir(path):
            # Folder
            output_folder = output_dir if output_dir else os.path.join(path, "grayscale_output")
            os.makedirs(output_folder, exist_ok=True)
            
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                        output_path = os.path.join(output_folder, f"{name}_gray{ext}")
                        if convert_to_grayscale(file_path, output_path):
                            processed += 1
    
    return processed

def main():
    print("=== Grayscale Image Converter ===\n")
    
    if len(sys.argv) > 1:
        # Command line arguments provided
        paths = sys.argv[1:]
        output_dir = None
    else:
        # Interactive mode
        print("Enter image file paths or folder paths (comma-separated):")
        input_str = input("> ").strip()
        paths = [p.strip() for p in input_str.split(',')]
        
        print("\nEnter output directory (press Enter to save alongside originals):")
        output_dir = input("> ").strip() or None
    
    total = process_images(paths, output_dir)
    print(f"\n✓ Successfully processed {total} images")

if __name__ == "__main__":
    main()
