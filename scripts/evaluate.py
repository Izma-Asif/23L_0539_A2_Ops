import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

from pipeline import _load_and_resize    

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", default="models/baseline_best.keras")
ap.add_argument("--mobilenet", default="models/mobilenet_stage2_best.keras")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--rule", choices=["f1", "youden"], default="f1")
a = ap.parse_args()

Path("reports").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
MODELS = {"Baseline CNN": a.baseline, "MobileNetV2 (stage 2)": a.mobilenet}


def get_probs(name, model_path, split):
    """Predict once per (model, split) and cache to CSV."""
    tag = name.split()[0].lower()
    cache = Path(f"reports/probs_{tag}_{split}.csv")
    df = pd.read_csv(f"splits/{split}.csv")
    if cache.exists():
        df["p_ai"] = pd.read_csv(cache)["p_ai"].values
        return df
    model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32")))
    ds = ds.map(lambda p, l: _load_and_resize(p, l, a.size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: tf.cast(x, tf.float32)).batch(32).prefetch(tf.data.AUTOTUNE)
    df["p_ai"] = model.predict(ds, verbose=0).reshape(-1)
    df[["path", "label", "p_ai"]].to_csv(cache, index=False)
    return df


def pick_threshold(y, p):
    if a.rule == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        return float(thr[np.argmax(tpr - fpr)])
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[np.argmax([f1_score(y, (p >= t).astype(int)) for t in ts])])


rows, rocs, cms, test_dfs = [], {}, {}, {}
for name, path in MODELS.items():
    val = get_probs(name, path, "val")
    thr = pick_threshold(val["label"], val["p_ai"])        # chosen on VAL only
    test = get_probs(name, path, "test")                    # test touched once
    y, p = test["label"].values, test["p_ai"].values
    pred = (p >= thr).astype(int)
    rows.append({"model": name, "threshold": round(thr, 3),
                 "accuracy": accuracy_score(y, pred),
                 "precision": precision_score(y, pred),
                 "recall": recall_score(y, pred),
                 "f1": f1_score(y, pred),
                 "roc_auc": roc_auc_score(y, p)})
    rocs[name] = roc_curve(y, p)
    cms[name] = confusion_matrix(y, pred)
    test["pred"] = pred
    test_dfs[name] = test

table = pd.DataFrame(rows).round(4)
table.to_csv("reports/test_metrics.csv", index=False)
print(table.to_string(index=False))

# ROC curves (both models)
plt.figure(figsize=(5, 5))
for name, (fpr, tpr, _) in rocs.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC {table.set_index('model').loc[name, 'roc_auc']})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves on the test set"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig8_roc.png", dpi=150); plt.close()

# Confusion matrices (both models)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["real", "AI"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig("figures/fig9_confusion.png", dpi=150); plt.close()

# 2.6 error analysis: 10 most confidently wrong test images (best model)
best = test_dfs["MobileNetV2 (stage 2)"].copy()
best["conf"] = np.where(best.pred == 1, best.p_ai, 1 - best.p_ai)
wrong = best[best.pred != best.label].sort_values("conf", ascending=False).head(10)
wrong.to_csv("reports/misclassified_top10.csv", index=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
    ax.imshow(Image.open(r["path"]).convert("RGB"))
    t, pr = ("AI" if r.label == 1 else "real"), ("AI" if r.pred == 1 else "real")
    ax.set_title(f"true: {t}\npred: {pr} ({r.conf:.2f})", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig("figures/fig10_errors.png", dpi=150); plt.close()
print("\nsaved fig8_roc.png, fig9_confusion.png, fig10_errors.png")
print(wrong[["path", "label", "pred", "conf"]].to_string(index=False))