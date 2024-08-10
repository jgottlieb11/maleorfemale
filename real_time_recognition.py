import cv2
import numpy as np
import pickle
from deepface import DeepFace
from src.face_detector import FaceDetector

def load_model(model_path="models/deepface_model.pkl"):
    with open(model_path, 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    return classifier, label_encoder

def recognize_faces_in_frame(classifier, label_encoder, frame, detector, confidence_threshold=0.2):
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        print("No faces detected.")
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        try:
            embedding = DeepFace.represent(rgb_face, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
        except Exception as e:
            print(f"Error in generating embedding: {e}")
            continue
        embedding = np.array(embedding).reshape(1, -1)
        prediction = classifier.predict(embedding)
        probabilities = classifier.predict_proba(embedding)[0]
        
        top_label_index = np.argmax(probabilities)
        top_label = label_encoder.inverse_transform([top_label_index])[0]
        top_probability = probabilities[top_label_index]
        
        if top_probability < confidence_threshold:
            print(f"Low confidence ({top_probability:.2f}). Skipping annotation.")
            continue

        print(f"Detected face with {top_label} ({top_probability:.2f})")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{top_label} ({top_probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def main():
    model_path = "models/deepface_model.pkl"
    classifier, label_encoder = load_model(model_path)
    detector = FaceDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = recognize_faces_in_frame(classifier, label_encoder, frame, detector)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
