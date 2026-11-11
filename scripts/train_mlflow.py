import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from pipeline import _load_and_resize          # same preprocessing as training

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--logging", choices=["manual", "auto"], default="manual")
ap.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--unfreeze", type=int, default=20)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--aug", choices=["on", "off"], default="on")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--local", action="store_true", help="log to local mlflow.db instead of DagsHub")
a = ap.parse_args()

if not a.local:
    load_dotenv()                     # MLFLOW_TRACKING_URI / USERNAME / PASSWORD from .env
SEED = 42
tf.keras.utils.set_random_seed(SEED)
L = tf.keras.layers
AUTOTUNE = tf.data.AUTOTUNE
out_dir = Path("reports/runs") / a.name
out_dir.mkdir(parents=True, exist_ok=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dvc_version():
    try:
        m = re.search(r"md5:\s*(\S+)", Path("data.dvc").read_text())
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


aug = tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(0.05),
                           L.RandomZoom(0.1), L.RandomContrast(0.1)])


def make_ds(split, train):
    df = pd.read_csv(f"splits/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32")))
    ds = ds.map(lambda p, l: _load_and_resize(p, l, a.size), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).cache()
    if train:
        ds = ds.shuffle(len(df), seed=SEED)
    ds = ds.batch(32)
    if train and a.aug == "on":                 # augmentation on training data only
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return df, ds.prefetch(AUTOTUNE)


_, train_ds = make_ds("train", True)
val_df, val_ds = make_ds("val", False)
test_df, test_ds = make_ds("test", False)

shape = (a.size, a.size, 3)
if a.backbone == "mobilenetv2":
    base = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet")
else:   # EfficientNetB0 rescales internally and expects raw 0-255 pixels
    base = tf.keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet")
base.trainable = False
if a.unfreeze > 0:
    for layer in base.layers[-a.unfreeze:]:
        layer.trainable = not isinstance(layer, L.BatchNormalization)

inputs = tf.keras.Input(shape)
x = inputs
if a.backbone == "mobilenetv2":
    x = L.Rescaling(1 / 127.5, offset=-1)(x)    # what MobileNetV2 weights expect
x = base(x, training=False)
x = L.GlobalAveragePooling2D()(x)
x = L.Dropout(a.dropout)(x)
outputs = L.Dense(1, activation="sigmoid", name="prob_ai")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(tf.keras.optimizers.Adam(a.lr), "binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)


class LogToMlflow(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
if a.logging == "manual":
    callbacks.append(LogToMlflow())
else:
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

mlflow.set_experiment("VisionGuard")
with mlflow.start_run(run_name=a.name) as run:
    if a.logging == "manual":
        mlflow.log_params({"backbone": a.backbone, "learning_rate": a.lr,
                           "unfrozen_layers": a.unfreeze, "dropout": a.dropout,
                           "augmentation": a.aug, "image_size": a.size,
                           "epochs_max": a.epochs, "batch_size": 32,
                           "optimizer": "adam", "trainable_params": n_trainable})
    mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_version(),
                     "seed": SEED, "logging_mode": a.logging, "backbone": a.backbone})

    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs,
                     callbacks=callbacks, verbose=2)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

    # threshold chosen on validation only, then applied to test
    pv = model.predict(val_ds, verbose=0).reshape(-1)
    pt = model.predict(test_ds, verbose=0).reshape(-1)
    yv, yt = val_df["label"].values, test_df["label"].values
    ts = np.linspace(0.05, 0.95, 91)
    thr = float(ts[np.argmax([f1_score(yv, (pv >= t).astype(int)) for t in ts])])
    pred = (pt >= thr).astype(int)

    metrics = {"best_epoch": best_epoch, "val_auc_final": float(roc_auc_score(yv, pv)),
               "val_loss_best": float(min(hist.history["val_loss"])), "threshold": thr,
               "test_accuracy": accuracy_score(yt, pred), "test_precision": precision_score(yt, pred),
               "test_recall": recall_score(yt, pred), "test_f1": f1_score(yt, pred),
               "test_auc": roc_auc_score(yt, pt)}
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    # artifacts (3.4)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=["real", "AI"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{a.name} (test)")
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fpr, tpr, _ = roc_curve(yt, pt)
    fig = plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC {metrics['test_auc']:.3f}"); plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title(a.name)
    fig.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "requirements_frozen.txt").write_text(freeze)

    for f in ["confusion_matrix.png", "roc_curve.png", "requirements_frozen.txt"]:
        mlflow.log_artifact(str(out_dir / f))
    for s in ["train", "val", "test"]:
        mlflow.log_artifact(f"splits/{s}.csv", artifact_path="splits")
    try:
        mlflow.keras.log_model(model, name="model")
    except Exception as e:
        print("log_model failed, logging the file instead:", repr(e)[:200])
        model.save(out_dir / "model.keras")
        mlflow.log_artifact(str(out_dir / "model.keras"))

    print(f"\nrun {a.name} ({a.logging}) id={run.info.run_id}")
    print({k: round(float(v), 4) for k, v in metrics.items()})