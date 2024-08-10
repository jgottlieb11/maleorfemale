import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(data_dir, save_model_path, subsample_fraction=0.01):
    print("Loading images and labels...")
    images = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)
                if label == "train":
                    continue
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(rgb_img)
                labels.append(label)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    if len(set(labels_encoded)) < 2:
        raise ValueError("The number of classes has to be greater than one; got 1 class")

    subsample_size = int(len(images) * subsample_fraction)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample_size, random_state=42)
    for train_index, test_index in sss.split(images, labels_encoded):
        images_subsampled = [images[i] for i in test_index]
        labels_subsampled = [labels_encoded[i] for i in test_index]

    print(f"Subsampled to {subsample_size} images.")

    label_distribution = {label: labels_subsampled.count(label) for label in set(labels_subsampled)}
    print("Label distribution after subsampling:", label_distribution)

    plt.bar(label_distribution.keys(), label_distribution.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution After Subsampling')
    plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)
    plt.show()

    X_train, X_val, y_train, y_val = train_test_split(images_subsampled, labels_subsampled, test_size=0.2, random_state=42)

    embeddings_train = []
    for img in tqdm(X_train, desc="Processing training images"):
        try:
            embedding = DeepFace.represent(img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
            embeddings_train.append(embedding)
        except:
            continue

    embeddings_val = []
    for img in tqdm(X_val, desc="Processing validation images"):
        try:
            embedding = DeepFace.represent(img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
            embeddings_val.append(embedding)
        except:
            continue

    print("Training the SVM classifier...")
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(embeddings_train, y_train)

    with open(save_model_path, 'wb') as f:
        pickle.dump((classifier, label_encoder), f)

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    data_dir = "data/train"
    save_model_path = "models/deepface_model.pkl"
    subsample_fraction = 0.1  # Use 10% of the data for testing- should be sufficient - increase this number of you would like a more "in depth" trained model

    train_model(data_dir, save_model_path, subsample_fraction)
