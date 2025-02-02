import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add the parent directory to the path so we can import from train/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from train.train import MoggingModel

class MoggingPredictor:
    def __init__(self, model_path='train/checkpoints/final_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MoggingModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms - same as validation transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict mogging probability for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            float: Probability of mogging (0-100%)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            prob = output.item() * 100  # Convert to percentage
            
        return prob
