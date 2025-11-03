# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Load the trained model once when the server starts
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_emotionModel.h5')
# model = load_model(MODEL_PATH)

# # Define the same class order you trained with
# class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# def predict_emotion(img_path):
#     """Takes an image path and returns the detected emotion label"""
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction)
#     return class_names[predicted_class]

# def map_to_message(emotion):
#     """Maps emotion to a friendly message and emoji"""
#     messages = {
#         "angry": ("Take a deep breath! It’s okay to feel angry sometimes.", "😠"),
#         "disgust": ("Yikes! Something clearly didn’t sit right.", "🤢"),
#         "fear": ("Don’t worry, you’re safe here.", "😨"),
#         "happy": ("Keep smiling! The world shines brighter with you.", "😄"),
#         "neutral": ("Nice and calm — balanced vibes.", "😐"),
#         "sad": ("It’s okay to feel sad. Brighter days are ahead.", "😢"),
#         "surprise": ("Whoa! Didn’t see that coming, huh?", "😲"),
#         "error": ("Error detecting emotion. Please try again.", "⚠️")
#     }
#     return messages.get(emotion, ("Unknown emotion", "❓"))

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

# Don't load model immediately
_model = None  

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def predict_emotion(img_path):
    model = get_model()  # load only when first used
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    emotion_index = int(np.argmax(prediction))
    return ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][emotion_index]


def map_to_message(emotion):
    messages = {
        'angry': ("Try to stay calm, you’ve got this!", "😡"),
        'disgust': ("Maybe step away for a bit.", "🤢"),
        'fear': ("It’s okay to be scared sometimes.", "😨"),
        'happy': ("Keep smiling! The world needs more of that!", "😄"),
        'neutral': ("Steady and composed. Nice balance!", "😐"),
        'sad': ("Cheer up — better days are ahead.", "😢"),
        'surprise': ("Wow! Didn’t see that coming!", "😲"),
    }
    return messages.get(emotion, ("Emotion not recognized", "❓"))

