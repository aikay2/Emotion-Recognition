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
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Locate the TensorFlow Lite model inside 'core' folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_emotionModel_quant.tflite')

# Global interpreter (cached so it doesn't reload on every request)
_interpreter = None
_input_details = None
_output_details = None


def get_interpreter():
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
    return _interpreter, _input_details, _output_details


def predict_emotion(img_path):
    interpreter, input_details, output_details = get_interpreter()

    # Preprocess image
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
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

