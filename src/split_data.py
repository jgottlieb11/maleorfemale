import os
import shutil
import random

def collect_images(source_dir):
    images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images

def split_data(source_dir, train_dir, val_dir, test_dir, val_split=0.1, test_split=0.1):
    
    images = collect_images(source_dir)
    print(f"Total images found: {len(images)}")

    random.shuffle(images)
    
    val_size = int(len(images) * val_split)
    test_size = int(len(images) * test_split)
    train_size = len(images) - val_size - test_size

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")

    for img in train_images:
        dest = os.path.join(train_dir, os.path.basename(img))
        shutil.copy(img, dest)
    for img in val_images:
        dest = os.path.join(val_dir, os.path.basename(img))
        shutil.copy(img, dest)
    for img in test_images:
        dest = os.path.join(test_dir, os.path.basename(img))
        shutil.copy(img, dest)

if __name__ == "__main__":
    source_dir = "data/train"  
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    split_data(source_dir, train_dir, val_dir, test_dir)
