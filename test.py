import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import cv2
import numpy as np
from PIL import Image

def test_setup():
    # Test CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test model loading
    try:
        processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224-22k")
        print("✓ Successfully loaded ConvNeXT model and processor")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    # Test camera
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Successfully accessed camera")
            else:
                print("✗ Could not read from camera")
        else:
            print("✗ Could not access camera")
        cap.release()
    except Exception as e:
        print(f"✗ Error accessing camera: {e}")

if __name__ == "__main__":
    print("Testing setup...")
    test_setup()
    print("\nSetup test complete!")