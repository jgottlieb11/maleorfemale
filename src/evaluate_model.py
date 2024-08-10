import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = 'models/deepface_model.pkl'
with open(model_path, 'rb') as f:
    classifier, label_encoder = pickle.load(f)

datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    directory='data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    subset='validation',
    shuffle=False
)

embeddings = []
labels = []
for batch in val_gen:
    images, batch_labels = batch
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(rgb_img, model_name='Facenet', detector_backend='opencv', enforce_detection=False)[0]['embedding']
        embeddings.append(embedding)
    labels.extend(batch_labels)
    if len(embeddings) >= val_gen.samples:
        break

embeddings = np.array(embeddings)
labels = np.array(labels)

labels_encoded = label_encoder.transform(labels)

accuracy = classifier.score(embeddings, labels_encoded)
print(f'Validation Accuracy: {accuracy}')
