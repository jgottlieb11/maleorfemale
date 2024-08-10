import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def fetch_data(img_dir="celeba/img_align_celeba/img_align_celeba", save_dir="data", img_shape=(100, 100)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bbox_data = pd.read_csv("celeba/list_bbox_celeba.csv")
    attr_data = pd.read_csv("celeba/list_attr_celeba.csv")

    print("Bounding Box Data:")
    print(bbox_data.head())
    print("Attributes Data:")
    print(attr_data.head())

    data = pd.merge(bbox_data, attr_data, on='image_id')

    print("Merged Data:")
    print(data.head())

    for _, row in tqdm(data.iterrows(), total=len(data)):
        img_path = os.path.join(img_dir, row['image_id'])
        
        print(f"Processing image: {img_path}")

        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to read image {img_path}")
            continue
        
        x, y, w, h = row['x_1'], row['y_1'], row['width'], row['height']
        face = image[y:y+h, x:x+w]
        
        if face.size == 0:
            print(f"Empty face image for {img_path} with bbox ({x}, {y}, {w}, {h})")
            continue

        face = cv2.resize(face, img_shape, interpolation=cv2.INTER_AREA)
        
        label = "Male" if row['Male'] == 1 else "Female"
        save_path = os.path.join(save_dir, label)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cv2.imwrite(os.path.join(save_path, row['image_id']), face)

if __name__ == "__main__":
    fetch_data(img_dir="celeba/img_align_celeba/img_align_celeba", save_dir="eval")
