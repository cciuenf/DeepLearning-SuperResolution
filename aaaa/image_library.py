"""
Image Library Management Module
Handles loading and organizing images from different directories
"""

import os
import cv2 as cv


class ImageLibrary:
    """Manages collections of images from different directories"""
    
    def __init__(self, directories):
        """
        Initialize the image library
        
        Args:
            directories (dict): Dictionary mapping directory names to paths
        """
        self.directories = directories
        self.image_library = {}
        self._load_all_images()
    
    def _load_all_images(self):
        """Load all images from the specified directories"""
        for dir_name in self.directories:
            self.image_library[dir_name] = {
                "lib_path": self.directories[dir_name],
                "images": {}
            }
        
        for dir_name in self.directories:
            if not os.path.exists(self.directories[dir_name]):
                print(f"Warning: Directory '{self.directories[dir_name]}' does not exist")
                continue
                
            filenames = os.listdir(self.directories[dir_name])
            sorted_filenames = sorted(filenames, key=lambda name: int(name.split('.')[0]) if name.split('.')[0].isdigit() else float('inf'))
            
            for file in sorted_filenames:
                file_path = os.path.join(self.image_library[dir_name]["lib_path"], file)
                cv_img = cv.imread(file_path)
                
                if cv_img is not None:
                    self.image_library[dir_name]["images"][file] = {
                        "path": file_path,
                        "cv_img": cv_img
                    }
                else:
                    print(f"Warning: Could not load image '{file_path}'")
    
    def get_images(self, lib_name):
        """
        Get all images from a specific library
        
        Args:
            lib_name (str): Name of the library/directory
            
        Returns:
            list: List of image dictionaries with 'path' and 'cv_img' keys
        """
        if lib_name not in self.image_library:
            return []
        return list(self.image_library[lib_name]["images"].values())
    
    def get_image_count(self, lib_name):
        """Get the number of images in a specific library"""
        return len(self.get_images(lib_name))
    
    def get_library_names(self):
        """Get all available library names"""
        return list(self.image_library.keys())
    
    def get_image_by_filename(self, lib_name, filename):
        """
        Get a specific image by filename
        
        Args:
            lib_name (str): Name of the library/directory
            filename (str): Name of the image file
            
        Returns:
            dict or None: Image dictionary or None if not found
        """
        if lib_name in self.image_library and filename in self.image_library[lib_name]["images"]:
            return self.image_library[lib_name]["images"][filename]
        return None
    
    def reload_library(self, lib_name):
        """Reload a specific library"""
        if lib_name in self.directories:
            self.image_library[lib_name] = {
                "lib_path": self.directories[lib_name],
                "images": {}
            }
            
            if os.path.exists(self.directories[lib_name]):
                filenames = os.listdir(self.directories[lib_name])
                sorted_filenames = sorted(filenames, key=lambda name: int(name.split('.')[0]) if name.split('.')[0].isdigit() else float('inf'))
                
                for file in sorted_filenames:
                    file_path = os.path.join(self.image_library[lib_name]["lib_path"], file)
                    cv_img = cv.imread(file_path)
                    
                    if cv_img is not None:
                        self.image_library[lib_name]["images"][file] = {
                            "path": file_path,
                            "cv_img": cv_img
                        }
