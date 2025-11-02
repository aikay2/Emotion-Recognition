# evaluate_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
MODEL_PATH = 'face_emotionModel.h5'     # path to your model
TEST_DIR = 'test'                   # path to your test folder
IMG_SIZE = (48, 48)                         # change if your model uses (64, 64) or others
COLOR_MODE = 'grayscale'                    # or 'rgb'
BATCH_SIZE = 32

# === Load model ===
print("Loading trained model...")
model = load_model(MODEL_PATH)

# === Prepare test data generator ===
datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === Predict ===
print("Evaluating on test data...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# === Accuracy ===
test_accuracy = np.mean(predicted_classes == true_classes)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%\n")

# === Detailed report ===
print("📊 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# === Confusion Matrix ===
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Data")
plt.show()
