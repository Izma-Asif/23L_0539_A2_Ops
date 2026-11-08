import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from models import (build_baseline_cnn, build_mobilenet, compile_model,
                    trainable_layer_names, unfreeze_last_block)
from pipeline import make_dataset

SEED = 42
for d in ("models", "reports", "figures"):
    Path(d).mkdir(exist_ok=True)


def callbacks(ckpt_path, patience=5):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def run_stage(model, train_ds, val_ds, epochs, ckpt_path, stage):
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=callbacks(ckpt_path), verbose=2)
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch_in_stage", range(1, len(df) + 1))
    df.insert(0, "stage", stage)
    best = int(df["val_loss"].idxmin()) + 1
    print(f"[{stage}] epochs run: {len(df)}  best epoch (lowest val_loss): {best}  "
          f"val_loss={df['val_loss'].min():.4f}")
    return df


def plot_history(df, path, title):
    df = df.reset_index(drop=True)
    df["epoch"] = df.index + 1                                    # global epoch counter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(df["epoch"], df[key], label="train")
        ax.plot(df["epoch"], df["val_" + key], label="validation")
        for stage, g in df.groupby("stage", sort=False):
            b = g.loc[g["val_loss"].idxmin()]                     # best epoch = lowest val_loss
            ax.scatter([b["epoch"]], [b["val_" + key]], marker="*", s=180, color="red", zorder=5,
                       label=f"best epoch ({stage}): {int(b['epoch'])}")
        stages = list(df["stage"].unique())
        if len(stages) > 1:
            ax.axvline(df[df["stage"] == stages[0]]["epoch"].max() + 0.5, color="black", ls=":")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(f"{title}: {label}")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("saved", path)


def save_trainable(model, path):
    names = trainable_layer_names(model)
    Path(path).write_text("\n".join(names))
    print(f"{path}: {len(names)} trainable layers; first/last: {names[:2]} ... {names[-3:]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    ap.add_argument("--epochs", type=int, default=30)      # baseline
    ap.add_argument("--epochs1", type=int, default=10)     # mobilenet stage 1 (frozen base)
    ap.add_argument("--epochs2", type=int, default=10)     # mobilenet stage 2 (last block unfrozen)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)                   # seeds Python, NumPy and TensorFlow
    train_ds = make_dataset("splits/train.csv", a.image_size, a.batch_size, training=True)
    val_ds = make_dataset("splits/val.csv", a.image_size, a.batch_size)

    if a.model == "baseline":
        model = build_baseline_cnn(a.image_size, learning_rate=a.lr1)
        save_trainable(model, "reports/trainable_baseline.txt")
        hist = run_stage(model, train_ds, val_ds, a.epochs, "models/baseline_best.keras", "baseline")
        hist.to_csv("reports/history_baseline.csv", index=False)
        plot_history(hist, "figures/fig6_baseline_curves.png", "Baseline CNN")
        print("val metrics:", dict(zip(model.metrics_names, model.evaluate(val_ds, verbose=0))))
    else:
        model, base = build_mobilenet(a.image_size, learning_rate=a.lr1)
        print(f"base layers: {len(base.layers)}")
        save_trainable(model, "reports/trainable_mobilenet_stage1.txt")
        h1 = run_stage(model, train_ds, val_ds, a.epochs1, "models/mobilenet_stage1_best.keras", "stage1_frozen")
        print("stage 1 val metrics:", dict(zip(model.metrics_names, model.evaluate(val_ds, verbose=0))))

        unfreeze_last_block(base)
        compile_model(model, a.lr2)                        # recompiling is required after changing trainable
        save_trainable(model, "reports/trainable_mobilenet_stage2.txt")
        h2 = run_stage(model, train_ds, val_ds, a.epochs2, "models/mobilenet_stage2_best.keras", "stage2_finetune")
        print("stage 2 val metrics:", dict(zip(model.metrics_names, model.evaluate(val_ds, verbose=0))))

        hist = pd.concat([h1, h2], ignore_index=True)
        hist.to_csv("reports/history_mobilenet.csv", index=False)
        plot_history(hist, "figures/fig7_mobilenet_curves.png", "MobileNetV2")