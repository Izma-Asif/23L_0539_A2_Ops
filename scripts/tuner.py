import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from pipeline import _load_and_resize      

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--size", type=int, default=224)
a = ap.parse_args()

tf.keras.utils.set_random_seed(42)
Path("reports").mkdir(exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE
L = tf.keras.layers
aug = tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(0.05),
                           L.RandomZoom(0.1), L.RandomContrast(0.1)])


def make_ds(split, train):
    df = pd.read_csv(f"splits/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32")))
    ds = ds.map(lambda p, l: _load_and_resize(p, l, a.size), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).cache()
    if train:
        ds = ds.shuffle(len(df), seed=42)
    ds = ds.batch(32)
    if train:          
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


train_ds, val_ds = make_ds("train", True), make_ds("val", False)


def build(hp):
    n = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(a.size, a.size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    if n > 0:
        for layer in base.layers[-n:]:
            layer.trainable = not isinstance(layer, L.BatchNormalization)
    inputs = tf.keras.Input((a.size, a.size, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
    out = L.Dense(1, activation="sigmoid", name="prob_ai")(x)
    model = tf.keras.Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-5, 1e-3, sampling="log")),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


tuner = kt.RandomSearch(build, objective="val_loss", max_trials=a.trials, seed=42,
                        directory="tuner_results", project_name="visionguard", overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=a.epochs, verbose=2,
             callbacks=[tf.keras.callbacks.EarlyStopping(
                 monitor="val_loss", patience=2, restore_best_weights=True)])

print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)
rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_loss": round(t.score, 4)}
        for t in tuner.oracle.get_best_trials(a.trials)]
table = pd.DataFrame(rows)
table.to_csv("reports/tuner_visionguard.csv", index=False)
print(table.to_string(index=False))

print("\nSpearman correlation with val_loss (closer to +/-1 = matters more):")
for col in ["lr", "dropout", "unfrozen_layers"]:
    print(f"  {col}: {table[col].corr(table['val_loss'], method='spearman'):+.2f}")