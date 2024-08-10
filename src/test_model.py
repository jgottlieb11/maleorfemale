import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'deepface_model.pkl')

with open(model_path, 'rb') as f:
    classifier, label_encoder = pickle.load(f)

img_path = 'celeba/img_align_celeba/img_align_celeba/202587.jpg'

image = cv2.imread(img_path)
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

embedding = DeepFace.represent(rgb_img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
embedding = np.array(embedding).reshape(1, -1)

prediction = classifier.predict(embedding)
predicted_label = label_encoder.inverse_transform(prediction)

print(f'Prediction: {predicted_label[0]}')
