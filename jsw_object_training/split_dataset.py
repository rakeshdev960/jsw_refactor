import os
import shutil
import random
from pathlib import Path

# Configuration
DATASET_DIR = "Combined_CementBag_Dataset"
TRAIN_RATIO = 0.8  # 80% for training, 20% for validation

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        f"{DATASET_DIR}/train/images",
        f"{DATASET_DIR}/train/labels",
        f"{DATASET_DIR}/valid/images",
        f"{DATASET_DIR}/valid/labels"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def split_dataset():
    """Split the dataset into training and validation sets."""
    # Get all image files
    all_images = list(Path(f"{DATASET_DIR}/train/images").glob("*.jpg"))
    random.shuffle(all_images)
    
    # Calculate split point
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Total images: {len(all_images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Move validation images and their corresponding labels
    for img_path in val_images:
        # Get corresponding label path
        label_path = Path(f"{DATASET_DIR}/train/labels/{img_path.stem}.txt")
        
        # Move image
        dest_img = Path(f"{DATASET_DIR}/valid/images/{img_path.name}")
        shutil.move(str(img_path), str(dest_img))
        
        # Move label if it exists
        if label_path.exists():
            dest_label = Path(f"{DATASET_DIR}/valid/labels/{label_path.name}")
            shutil.move(str(label_path), str(dest_label))

def update_data_yaml():
    """Update the data.yaml file with correct paths."""
    content = f"""train: {DATASET_DIR}/train/images
val: {DATASET_DIR}/valid/images

nc: 1
names: ['cement bag']
"""
    with open(f"{DATASET_DIR}/data.yaml", "w") as f:
        f.write(content)

if __name__ == "__main__":
    print("Starting dataset split...")
    create_directories()
    split_dataset()
    update_data_yaml()
    print("Dataset split completed successfully!") 