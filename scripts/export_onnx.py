import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from pipeline import _load_and_resize          # the exact training-time preprocessing

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/mobilenet_stage2_best.keras")
ap.add_argument("--out", default="api/model.onnx")
ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()

out = Path(a.out)
out.parent.mkdir(exist_ok=True)
model = tf.keras.models.load_model(a.model, compile=False)

# FIX: Keras refuses to export a model that was loaded but never run.
# One dummy forward pass "calls" it so export knows the input shape.
_shape = [d or 224 for d in model.input_shape[1:]]
model(np.zeros((1, *_shape), dtype="float32"))

model.export(str(out), format="onnx", verbose=False)
print(f"exported {out}  ({out.stat().st_size / 1e6:.1f} MB)")

sys.path.insert(0, str(out.parent))
api = importlib.import_module("app")            # api/app.py: loads model.onnx, has the API's preprocess()

df = pd.read_csv("splits/val.csv").sample(a.n, random_state=42)
rows = []
for path, label in zip(df["path"], df["label"]):
    img, _ = _load_and_resize(tf.constant(path), tf.constant(float(label)), 224)
    p_keras = float(model.predict(tf.cast(img, tf.float32)[None].numpy(), verbose=0)[0, 0])
    x = api.preprocess(Path(path).read_bytes())
    p_onnx = float(api.session.run(None, {api.INPUT_NAME: x})[0].reshape(-1)[0])
    rows.append({"path": path, "p_keras": p_keras, "p_onnx_api": p_onnx, "abs_diff": abs(p_keras - p_onnx)})

res = pd.DataFrame(rows)
res.to_csv("reports/onnx_parity.csv", index=False)
flips = int(((res.p_keras >= 0.5) != (res.p_onnx_api >= 0.5)).sum())
print(f"images compared: {len(res)}")
print(f"max |diff| = {res.abs_diff.max():.4f}   mean |diff| = {res.abs_diff.mean():.4f}")
print(f"label flips at threshold 0.5: {flips}")
if res.abs_diff.max() > 0.05:
    print("WARNING: difference above 0.05 - check preprocessing before deploying")