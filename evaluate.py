import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'D:/test/CAPSTONE GITAM/pneumonia_dataset_kaggle/chest_xray/chest_xray'
MODEL_PATH = 'best_pneumonia_finetuned.keras'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    f'{DATA_DIR}/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  
)

print("Test data loaded successfully!")


print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
