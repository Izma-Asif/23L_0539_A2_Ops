import tensorflow as tf

IMAGE_SIZE = 224


def build_baseline_cnn(image_size=IMAGE_SIZE, dropout=0.4, learning_rate=1e-3):
    """Small CNN trained from scratch. Input: float32 pixels in [0, 255]; rescaling is inside."""
    L = tf.keras.layers
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = L.Rescaling(1.0 / 255, name="rescale")(inputs)

    for i, filters in enumerate([32, 64, 128], start=1):
        x = L.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = L.BatchNormalization(name=f"bn{i}")(x)
        x = L.MaxPooling2D(2, name=f"pool{i}")(x)

    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    outputs = L.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model