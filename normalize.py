import os
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import shutil

class ImageNormalizer:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the normalizer with target image size.
        Default size 224x224 matches ConvNeXT's expected input.
        """
        self.target_size = target_size
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_output_directory(self, input_dir):
        """Create normalized output directory if it doesn't exist."""
        input_path = Path(input_dir)
        output_dir = input_path.parent / f"{input_path.name}_normalized"
        
        if output_dir.exists():
            self.logger.warning(f"Output directory {output_dir} already exists. Cleaning...")
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True)
        return output_dir

    def normalize_image(self, image_path):
        """
        Normalize a single image:
        1. Load image
        2. Resize to target size while maintaining aspect ratio
        3. Convert to RGB if necessary
        """
        try:
            # Open image with PIL (handles more formats than cv2)
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate aspect ratio preserving dimensions
            aspect_ratio = image.size[0] / image.size[1]
            if aspect_ratio > 1:
                # Width is greater than height
                new_width = self.target_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                # Height is greater than width
                new_height = self.target_size[1]
                new_width = int(new_height * aspect_ratio)
            
            # Resize image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with padding
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            paste_x = (self.target_size[0] - new_width) // 2
            paste_y = (self.target_size[1] - new_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            return new_image
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self, input_dir):
        """
        Process all images in the input directory and save normalized versions.
        """
        input_dir = Path(input_dir)
        output_dir = self.setup_output_directory(input_dir)
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Process each image
        total_images = 0
        successful_images = 0
        
        for image_file in input_dir.rglob('*'):
            if image_file.suffix.lower() in valid_extensions:
                total_images += 1
                self.logger.info(f"Processing: {image_file.name}")
                
                # Normalize image
                normalized_image = self.normalize_image(image_file)
                
                if normalized_image:
                    # Save as JPEG
                    output_path = output_dir / f"{image_file.stem}.jpg"
                    normalized_image.save(output_path, 'JPEG', quality=95)
                    successful_images += 1
        
        # Log summary
        self.logger.info(f"\nNormalization complete!")
        self.logger.info(f"Total images processed: {total_images}")
        self.logger.info(f"Successfully normalized: {successful_images}")
        self.logger.info(f"Failed: {total_images - successful_images}")
        self.logger.info(f"Normalized images saved to: {output_dir}")

def main():
    # Example usage
    normalizer = ImageNormalizer(target_size=(224, 224))
    
    # Get input directory from user
    input_dir = input("Enter the path to your image directory: ").strip()
    
    if not os.path.exists(input_dir):
        print("Error: Directory does not exist!")
        return
    
    normalizer.process_directory(input_dir)

if __name__ == "__main__":
    main()