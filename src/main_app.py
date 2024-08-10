import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from face_detector import FaceDetector

def load_model(model_path="models/deepface_model.pkl"):
    with open(model_path, 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    return classifier, label_encoder

def recognize_faces(classifier, label_encoder, image, detector):
    faces = detector.detect_faces(image)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(rgb_face, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
        embedding = np.array(embedding).reshape(1, -1)
        prediction = classifier.predict(embedding)
        probabilities = classifier.predict_proba(embedding)[0]
        
        top_label_index = np.argmax(probabilities)
        second_label_index = np.argsort(probabilities)[-2]
        top_label = label_encoder.inverse_transform([top_label_index])[0]
        second_label = label_encoder.inverse_transform([second_label_index])[0]
        top_probability = probabilities[top_label_index]
        second_probability = probabilities[second_label_index]

        label = second_label 
        confidence = second_probability

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

if __name__ == "__main__":
    model_path = "models/deepface_model.pkl"
    classifier, label_encoder = load_model(model_path)
    detector = FaceDetector()

    image_path = "sample/test.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = recognize_faces(classifier, label_encoder, image, detector)
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
