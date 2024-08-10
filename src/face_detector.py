import cv2

class FaceDetector:
    def __init__(self, model_path="src/face_detector.xml"):
        self.face_cascade = cv2.CascadeClassifier(model_path)

    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        return faces

    def draw_faces(self, image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image

