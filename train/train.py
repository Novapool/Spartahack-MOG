import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from transformers import ConvNextModel
import torch.nn as nn
import torch.nn.functional as F
import yaml

class MoggingDataset(Dataset):
    """Dataset class for mogging classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all image files and labels
        self.samples = []
        
        # Mogging images (label 1)
        mogging_path = self.data_dir / 'mogging'
        for img_path in mogging_path.glob('*.jpg'):
            self.samples.append((img_path, 1))
            
        # Not mogging images (label 0)
        not_mogging_path = self.data_dir / 'not_mogging'
        for img_path in not_mogging_path.glob('*.jpg'):
            self.samples.append((img_path, 0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

class MoggingModel(pl.LightningModule):
    """PyTorch Lightning module for mogging classification"""
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        
        # Load pretrained ConvNeXT as base
        self.base_model = ConvNextModel.from_pretrained('facebook/convnext-base-224-22k')
        
        # Freeze most layers
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers for fine-tuning
        for param in self.base_model.encoder.stages[-1].parameters():
            param.requires_grad = True
        
        # Add pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
            
        # Add classification head
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),  # ConvNeXT base outputs 1024 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x):
        # Get features from base model
        features = self.base_model(x).last_hidden_state  # Shape: [batch_size, channels, height, width]
        
        # Global average pooling
        features = self.pool(features)  # Shape: [batch_size, channels, 1, 1]
        features = features.flatten(1)  # Shape: [batch_size, channels]
        
        # Classification
        logits = self.classification_head(features)
        return torch.sigmoid(logits)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat.squeeze(), y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy(y_hat.squeeze(), y)
        
        # Calculate accuracy
        preds = (y_hat.squeeze() > 0.5).float()
        acc = (preds == y).float().mean()
        
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

def get_transforms():
    """Get image transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(config_path='config.yml'):
    """Main training function"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = MoggingDataset(
        config['train_dir'],
        transform=train_transform
    )
    val_dataset = MoggingDataset(
        config['val_dir'],
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = MoggingModel(learning_rate=config['learning_rate'])
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['checkpoint_dir'],
            filename='mogging-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}',
            monitor='val_acc',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            mode='min'
        )
    ]
    
    # Initialize trainer with GPU support
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',  # Enable GPU
        devices=1,          # Use 1 GPU
        callbacks=callbacks,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'final_model.pth'))

if __name__ == "__main__":
    train_model()