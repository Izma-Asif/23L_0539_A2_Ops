import pandas as pd
import tensorflow as tf
import keras_tuner as kt

tf.keras.utils.set_random_seed(42)
(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x, y = x[:20000] / 255.0, y[:20000]        
x_tr, y_tr, x_val, y_val = x[:-4000], y[:-4000], x[-4000:], y[-4000:]

def build(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6, seed=42,
                        directory="tuner_results", project_name="toy_fashion", overwrite=True)
tuner.search(x_tr, y_tr, epochs=3, validation_data=(x_val, y_val), verbose=2)

print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)
rows = [{**t.hyperparameters.values, "val_accuracy": round(t.score, 4)}
        for t in tuner.oracle.get_best_trials(6)]
table = pd.DataFrame(rows)
table.to_csv("reports/tuner_toy.csv", index=False)
print(table.to_string(index=False))