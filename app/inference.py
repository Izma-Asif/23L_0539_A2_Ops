import os

import tensorflow as tf

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.keras")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.33"))     
MODEL_NAME = os.environ.get("MODEL_NAME", "effnet_b0_v2")


model = tf.keras.models.load_model(MODEL_PATH, compile=False)
SIZE = model.input_shape[1]


def preprocess(image_bytes: bytes):
  
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (SIZE, SIZE), method="bilinear", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return tf.cast(img, tf.float32)[None]


def predict(image_bytes: bytes):
    
    p_ai = float(model(preprocess(image_bytes), training=False)[0, 0])
    if p_ai >= THRESHOLD:
        return "AI-generated", p_ai
    return "real", 1.0 - p_ai


if __name__ == "__main__":         
    import sys
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            label, conf = predict(f.read())
        print(f"{path}: {label} (confidence {conf:.2f})")