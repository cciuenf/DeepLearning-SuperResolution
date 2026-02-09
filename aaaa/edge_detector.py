from PIL import Image, ImageFilter
import os
import sys

def detect_edges(input_path, output_path, method='canny', threshold1=100, threshold2=200):
    """
    Apply edge detection to a single image
    
    Methods:
    - 'canny': Canny edge detection (requires cv2)
    - 'sobel': Sobel edge detection (requires cv2)
    - 'find_edges': PIL's built-in edge detection
    - 'contour': PIL's contour filter
    """
    try:
        img = Image.open(input_path)
        
        if method in ['canny', 'sobel']:
            try:
                import cv2
                import numpy as np
                
                img_array = np.array(img.convert('RGB'))
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                if method == 'canny':
                    edges = cv2.Canny(gray, threshold1, threshold2)
                elif method == 'sobel':
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    edges = np.sqrt(sobelx**2 + sobely**2)
                    edges = np.uint8(edges / edges.max() * 255)
                
                result_img = Image.fromarray(edges)
                
            except ImportError:
                print(f"⚠ OpenCV not installed. Falling back to PIL's FIND_EDGES filter.")
                result_img = img.convert('L').filter(ImageFilter.FIND_EDGES)
        
        elif method == 'find_edges':
            result_img = img.convert('L').filter(ImageFilter.FIND_EDGES)
        
        elif method == 'contour':
            result_img = img.convert('L').filter(ImageFilter.CONTOUR)
        
        else:
            print(f"⚠ Unknown method '{method}'. Using 'find_edges'.")
            result_img = img.convert('L').filter(ImageFilter.FIND_EDGES)
        
        result_img.save(output_path)
        print(f"✓ Processed: {os.path.basename(input_path)} (method={method})")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def process_images(paths, method='find_edges', threshold1=100, threshold2=200, output_dir=None):
    """Process multiple images or folders"""
    processed = 0
    
    for path in paths:
        if os.path.isfile(path):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_edges{ext}")
            else:
                name, ext = os.path.splitext(path)
                output_path = f"{name}_edges{ext}"
            
            if detect_edges(path, output_path, method, threshold1, threshold2):
                processed += 1
                
        elif os.path.isdir(path):
            output_folder = output_dir if output_dir else os.path.join(path, "edges_output")
            os.makedirs(output_folder, exist_ok=True)
            
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                        output_path = os.path.join(output_folder, f"{name}_edges{ext}")
                        if detect_edges(file_path, output_path, method, threshold1, threshold2):
                            processed += 1
    
    return processed

def main():
    print("=== Edge Detection Image Processor ===\n")
    
    if len(sys.argv) > 1:
        # Command line arguments
        if sys.argv[1] in ['canny', 'sobel', 'find_edges', 'contour']:
            method = sys.argv[1]
            paths = sys.argv[2:]
        else:
            method = 'find_edges'
            paths = sys.argv[1:]
        
        threshold1 = 100
        threshold2 = 200
        output_dir = None
    else:
        # Interactive mode
        print("Enter image file paths or folder paths (comma-separated):")
        input_str = input("> ").strip()
        paths = [p.strip() for p in input_str.split(',')]
        
        print("\nChoose edge detection method:")
        print("  1. find_edges (PIL, fast, no dependencies)")
        print("  2. contour (PIL, simple contour detection)")
        print("  3. canny (OpenCV, best quality, requires cv2)")
        print("  4. sobel (OpenCV, gradient-based, requires cv2)")
        method_choice = input("Enter choice (1-4, default=1): ").strip()
        
        method_map = {
            '1': 'find_edges',
            '2': 'contour',
            '3': 'canny',
            '4': 'sobel',
            '': 'find_edges'
        }
        method = method_map.get(method_choice, 'find_edges')
        
        threshold1 = 100
        threshold2 = 200
        
        if method in ['canny', 'sobel']:
            print("\nCanny thresholds (default: 100, 200):")
            t1_input = input("Lower threshold (default=100): ").strip()
            t2_input = input("Upper threshold (default=200): ").strip()
            threshold1 = int(t1_input) if t1_input else 100
            threshold2 = int(t2_input) if t2_input else 200
        
        print("\nEnter output directory (press Enter to save alongside originals):")
        output_dir = input("> ").strip() or None
    
    total = process_images(paths, method, threshold1, threshold2, output_dir)
    print(f"\n✓ Successfully processed {total} images")
    
    if method in ['canny', 'sobel']:
        try:
            import cv2
        except ImportError:
            print("\n💡 Tip: Install OpenCV for better edge detection:")
            print("   pip install opencv-python")

if __name__ == "__main__":
    main()
