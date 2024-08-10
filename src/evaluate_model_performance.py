import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from deepface import DeepFace
from tqdm import tqdm
from face_detector import FaceDetector

def load_model(model_path="models/deepface_model.pkl"):
    with open(model_path, 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    return classifier, label_encoder

def load_labels(csv_file):
    
    df = pd.read_csv(csv_file)
    
    df['label'] = df['Male'].apply(lambda x: 'Male' if x == 1 else 'Female')
    return df.set_index('image_id')['label'].to_dict()

def evaluate_model_on_images(model, label_encoder, image_dir, labels_mapping, num_images=2000):
    y_true = []
    y_pred = []
    detector = FaceDetector()
    images_processed = 0

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg") and images_processed < num_images:
                img_path = os.path.join(root, file)
                label = labels_mapping.get(file)
                
                if label is None:
                    print(f"Skipping file with no label: {file}")
                    continue
                
                if label not in label_encoder.classes_:
                    print(f"Skipping invalid label: {label}")
                    continue

                y_true.append(label)

                image = cv2.imread(img_path)
                faces = detector.detect_faces(image)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = image[y:y+h, x:x+w]
                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    embedding = DeepFace.represent(rgb_face, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
                    embedding = np.array(embedding).reshape(1, -1)
                    prediction = model.predict(embedding)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    y_pred.append(predicted_label)
                else:
                    y_pred.append("Unknown")

                images_processed += 1

    print(f"Total images processed: {images_processed}")
    print(f"y_true labels: {set(y_true)}")
    print(f"y_pred labels: {set(y_pred)}")

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Face Recognition')
    plt.show()

def main():
    model_path = "models/deepface_model.pkl"
    image_dir = "data/train"
    
    labels_mapping = load_labels("celeba/list_attr_celeba.csv")
    
    classifier, label_encoder = load_model(model_path)

    y_true, y_pred = evaluate_model_on_images(classifier, label_encoder, image_dir, labels_mapping, num_images=8000)

    labels = label_encoder.classes_  # ['Female', 'Male']

    plot_confusion_matrix(y_true, y_pred, labels)

if __name__ == "__main__":
    main()
