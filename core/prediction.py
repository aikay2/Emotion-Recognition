import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model once when the server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_emotionModel.h5')
model = load_model(MODEL_PATH)

# Define the same class order you trained with
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(img_path):
    """Takes an image path and returns the detected emotion label"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

def map_to_message(emotion):
    """Maps emotion to a friendly message and emoji"""
    messages = {
        "angry": ("Take a deep breath! It’s okay to feel angry sometimes.", "😠"),
        "disgust": ("Yikes! Something clearly didn’t sit right.", "🤢"),
        "fear": ("Don’t worry, you’re safe here.", "😨"),
        "happy": ("Keep smiling! The world shines brighter with you.", "😄"),
        "neutral": ("Nice and calm — balanced vibes.", "😐"),
        "sad": ("It’s okay to feel sad. Brighter days are ahead.", "😢"),
        "surprise": ("Whoa! Didn’t see that coming, huh?", "😲"),
        "error": ("Error detecting emotion. Please try again.", "⚠️")
    }
    return messages.get(emotion, ("Unknown emotion", "❓"))
