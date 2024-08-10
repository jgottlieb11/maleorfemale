import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model_path = 'models/deepface_model.pkl'
with open(model_path, 'rb') as f:
    classifier, label_encoder = pickle.load(f)

datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    directory='data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

predictions = []
for i in range(len(val_gen)):
    images, labels = val_gen[i]
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(rgb_img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
        embedding = np.array(embedding).reshape(1, -1)
        prediction = classifier.predict(embedding)
        predictions.append(prediction[0])

predictions = np.array(predictions).flatten()

plt.hist(predictions, bins=20, edgecolor='black')
plt.xlabel('Prediction Label')
plt.ylabel('Frequency')
plt.title('Histogram of Predictions')
plt.show()
