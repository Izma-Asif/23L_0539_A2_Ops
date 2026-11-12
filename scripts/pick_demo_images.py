import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import inference
from pipeline import _load_and_resize

df = pd.read_csv("splits/test.csv")


diffs = []
for p in df["path"].head(20):
    train_img, _ = _load_and_resize(tf.constant(p), tf.constant(0.0), inference.SIZE)
    serve_img = inference.preprocess(Path(p).read_bytes())[0]
    diffs.append(float(np.abs(train_img.numpy().astype("float32") - serve_img.numpy()).max()))
print(f"preprocessing parity on 20 images: max pixel difference = {max(diffs):.3f} (0-255 scale)")
rows = []
for p, y in zip(df["path"], df["label"]):
    label, conf = inference.predict(Path(p).read_bytes())
    correct = (label == "AI-generated") == (int(y) == 1)
    rows.append({"path": p, "true": "AI" if int(y) == 1 else "real", "pred": label,
                 "confidence": round(conf, 3), "correct": correct})
res = pd.DataFrame(rows)
print(f"\nchampion accuracy on test at threshold {inference.THRESHOLD}: {res.correct.mean():.3f}")
for cls in ["real", "AI"]:
    top = res[(res.true == cls) & res.correct].sort_values("confidence", ascending=False).head(4)
    print(f"\nmost confident correct '{cls}' images:")
    print(top[["path", "pred", "confidence"]].to_string(index=False))