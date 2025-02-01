import os
import shutil
import random
from pathlib import Path

def organize_data(mogging_dir, not_mogging_dir, output_dir, val_split=0.2):
    """
    Organize data into train and validation sets.
    
    Args:
        mogging_dir: Directory containing mogging images
        not_mogging_dir: Directory containing not_mogging images
        output_dir: Where to create the organized dataset
        val_split: Fraction of data to use for validation
    """
    # Create directory structure
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        for label in ['mogging', 'not_mogging']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    # Function to split and copy files
    def process_directory(src_dir, label):
        # Get all jpg files
        files = list(Path(src_dir).glob('*.jpg'))
        random.shuffle(files)
        
        # Split into train and validation
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, output_path / 'train' / label / f.name)
        for f in val_files:
            shutil.copy2(f, output_path / 'val' / label / f.name)
            
        return len(train_files), len(val_files)
    
    # Process both directories
    print("Organizing dataset...")
    mog_train, mog_val = process_directory(mogging_dir, 'mogging')
    not_train, not_val = process_directory(not_mogging_dir, 'not_mogging')
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Training set:")
    print(f"  Mogging:     {mog_train} images")
    print(f"  Not Mogging: {not_train} images")
    print(f"Validation set:")
    print(f"  Mogging:     {mog_val} images")
    print(f"  Not Mogging: {not_val} images")

if __name__ == "__main__":
    # Get paths from user
    mogging_dir = input("Enter path to normalized mogging images: ").strip()
    not_mogging_dir = input("Enter path to normalized not_mogging images: ").strip()
    output_dir = input("Enter path for organized dataset: ").strip()
    
    # Validate paths
    if not all(os.path.exists(d) for d in [mogging_dir, not_mogging_dir]):
        print("Error: One or more input directories don't exist!")
        exit(1)
        
    organize_data(mogging_dir, not_mogging_dir, output_dir)