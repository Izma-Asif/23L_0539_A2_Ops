import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--remote", action="store_true", help="log to DagsHub (reads .env)")
a = ap.parse_args()

if a.remote:
    from dotenv import load_dotenv
    load_dotenv()          

tf.keras.utils.set_random_seed(42)
(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data()
x, y, xt, yt = x[:10000] / 255.0, y[:10000], xt[:2000] / 255.0, yt[:2000]


class LogToMlflow(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            mlflow.log_metric(k, float(v), step=epoch)


mlflow.set_experiment("toy_mnist")
with mlflow.start_run(run_name=f"lr_{a.lr}"):
    mlflow.log_params({"learning_rate": a.lr, "epochs": a.epochs, "hidden_units": 64})
    mlflow.set_tag("purpose", "tool lab")

    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(tf.keras.optimizers.Adam(a.lr),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(x, y, epochs=a.epochs, validation_split=0.2,
                     verbose=2, callbacks=[LogToMlflow()])

    _, test_acc = model.evaluate(xt, yt, verbose=0)
    mlflow.log_metric("test_accuracy", float(test_acc))

    fig = plt.figure()
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.legend(); plt.title(f"lr={a.lr}")
    mlflow.log_figure(fig, "loss_curve.png")
    plt.close(fig)

    try:
        mlflow.keras.log_model(model, name="model")
    except Exception as e:                       
        print("log_model failed, logging the file instead:", repr(e)[:200])
        model.save("toy_model.keras")
        mlflow.log_artifact("toy_model.keras")
print("done")