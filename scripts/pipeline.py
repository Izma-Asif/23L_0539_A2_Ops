from pathlib import Path
import pandas as pd
import tensorflow as tf

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter(seed=SEED):
    """Random augmentation, used for the TRAINING set only."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed),     # +-5% of a full turn (~18 degrees)
        tf.keras.layers.RandomZoom(0.10, seed=seed),
        tf.keras.layers.RandomContrast(0.10, seed=seed),
    ], name="augmentation")


def _load_and_resize(path, label, image_size):
    raw = tf.io.read_file(path)
    # channels=3 forces RGB: greyscale is replicated, palette is expanded, alpha is dropped.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    # Plain resize: stretches to a square (aspect ratio NOT preserved). antialias avoids aliasing
    # when shrinking the very large originals.
    img = tf.image.resize(img, (image_size, image_size), method="bilinear", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)   # uint8 keeps the cache small
    img.set_shape((image_size, image_size, 3))
    return img, label


def make_dataset(csv_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=False,
                 cache=True, seed=SEED, num_parallel_calls=4):
    """Build a batched tf.data pipeline from a split CSV (columns: path, label).

    Yields (images float32 in [0, 255], shape [B, S, S, 3]; labels float32 [B]).
    Rescaling / preprocess_input is deliberately NOT done here: it lives inside the model so
    training and serving use exactly the same code.
    """
    df = pd.read_csv(csv_path)
    paths = df["path"].astype(str).values
    labels = df["label"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _load_and_resize(p, y, image_size), num_parallel_calls=num_parallel_calls)
    if cache:
        ds = ds.cache()                      # decode + resize only once (first epoch)
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(SEED)
    Path("figures").mkdir(exist_ok=True)

    train_ds = make_dataset("splits/train.csv", training=True)
    val_ds = make_dataset("splits/val.csv")
    test_ds = make_dataset("splits/test.csv")
    print("element_spec (train):", train_ds.element_spec)
    print("batches  train/val/test:", len(train_ds), len(val_ds), len(test_ds))

    images, labels = next(iter(train_ds))
    print("one training batch:", images.shape, images.dtype, "labels", labels.shape)
    print(f"pixel range after augmentation: {float(tf.reduce_min(images)):.1f} .. {float(tf.reduce_max(images)):.1f}")
    print("labels in this batch (1 = AI-generated):", int(tf.reduce_sum(labels)), "of", len(labels))

    names = {0.0: "real", 1.0: "AI"}
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, img, y in zip(axes.flat, images[:9], labels[:9]):
        ax.imshow(tf.clip_by_value(img, 0, 255).numpy().astype("uint8"))
        ax.set_title(names[float(y)], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch (first 9 of {len(labels)}, {IMAGE_SIZE}x{IMAGE_SIZE})")
    fig.tight_layout()
    fig.savefig("figures/fig5_augmented_batch.png", dpi=150)
    print("Saved figures/fig5_augmented_batch.png")