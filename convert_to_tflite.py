import tensorflow as tf
import numpy as np

h5_model_path = "face_emotionModel.h5"
tflite_model_path = "face_emotionModel_quant.tflite"

# Load model
model = tf.keras.models.load_model(h5_model_path)

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# (Optional but recommended) Provide a representative dataset for better accuracy
def representative_data_gen():
    for _ in range(100):
        dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
        yield [dummy_input]

converter.representative_dataset = representative_data_gen

# Force full integer quantization (saves more space)
converter.target_spec.supported_types = [tf.float16]

# Convert model
tflite_model = converter.convert()

# Save the new quantized model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Quantized TensorFlow Lite model saved as:", tflite_model_path)
