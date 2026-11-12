import argparse
from pathlib import Path

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

from pipeline import _load_and_resize      # same preprocessing as training

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="data/RealArt/RealArt/wildlife_hero.jpg")
ap.add_argument("--threshold", type=float, default=0.33)   # chosen on validation for effnet_b0
a = ap.parse_args()

load_dotenv()
mv = MlflowClient().get_model_version_by_alias("VisionGuard", "champion")
print(f"alias 'champion' -> version {mv.version} (run {mv.run_id})")
print("source:", mv.source)

Path("models/_download").mkdir(parents=True, exist_ok=True)
folder = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path="models/_download")
files = list(Path(folder).rglob("*.keras"))
if not files:
    print("no .keras file found, folder contains:")
    for f in Path(folder).rglob("*"):
        print("  ", f)
    raise SystemExit(1)
model = tf.keras.models.load_model(files[0], compile=False)

model.save("models/champion.keras")
print("saved local copy: models/champion.keras")

size = model.input_shape[1]
img, _ = _load_and_resize(tf.constant(a.image), tf.constant(0.0), size)
p = float(model.predict(tf.cast(img, tf.float32)[None], verbose=0)[0, 0])
label = "AI-generated" if p >= a.threshold else "real"
print(f"image: {a.image}")
print(f"probability AI = {p:.3f}  threshold = {a.threshold}  ->  {label}")