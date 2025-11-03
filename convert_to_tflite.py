import tensorflow as tf
from tensorflow.keras.models import model_from_json
import h5py

h5_model_path = "face_emotionModel.h5"
tflite_model_path = "face_emotionModel_quant.tflite"

# --- STEP 1: Read model config manually ---
with h5py.File(h5_model_path, 'r') as f:
    model_config = f.attrs.get('model_config')
    if model_config is None:
        raise ValueError("No model config found inside .h5 file")
    # If it's bytes, decode. If it's already a string, keep it.
    if isinstance(model_config, bytes):
        model_json = model_config.decode('utf-8')
    else:
        model_json = model_config
    model = model_from_json(model_json)

# --- STEP 2: Load weights ---
model.load_weights(h5_model_path)

# --- STEP 3: Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Successfully converted to TFLite:", tflite_model_path)
