import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def extract_embeddings(image_dir):
    embeddings = []
    labels = []
    valid_classes = {'Female', 'Male', 'train'}
    
    for root, dirs, files in os.walk(image_dir):
        class_name = os.path.basename(root)
        if class_name not in valid_classes:
            continue
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                embedding = DeepFace.represent(rgb_img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
                embeddings.append(embedding)
                labels.append(class_name)
    
    return np.array(embeddings), labels

def train_svm_model(embeddings, labels, save_model_path="models/deepface_model.pkl"):
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=42)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    with open(save_model_path, 'wb') as f:
        pickle.dump((classifier, label_encoder), f)

    print("Model trained and saved successfully.")
    
    val_accuracy = classifier.score(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy}')

if __name__ == "__main__":
    image_dir = "data/train" 
    embeddings, labels = extract_embeddings(image_dir)
    train_svm_model(embeddings, labels)
