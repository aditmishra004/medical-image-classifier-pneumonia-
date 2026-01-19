import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

IMAGE_SIZE = (224, 224)
MODEL_PATH = 'best_pneumonia_finetuned.keras'

if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    print(f"Loading image from: {image_path}")
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
except Exception as e:
    print(f"Error loading model or image: {e}")
    sys.exit(1)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array /= 255.0  
prediction = model.predict(img_array)
probability = prediction[0][0]

print("\n--- Prediction Result ---")
if probability > 0.5:
    print(f"Prediction: PNEUMONIA")
    print(f"Confidence: {probability * 100:.2f}%")
else:
    print(f"Prediction: NORMAL")
    print(f"Confidence: {(1 - probability) * 100:.2f}%")
