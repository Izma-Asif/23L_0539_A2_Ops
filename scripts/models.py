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


def build_mobilenet(image_size=IMAGE_SIZE, dropout=0.3, learning_rate=1e-3):
    """MobileNetV2 (ImageNet weights, no classifier) frozen + our own head. Returns (model, base)."""
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    base.trainable = False                                    # stage 1: base frozen

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    # Same maths as mobilenet_v2.preprocess_input ([0,255] -> [-1,1]), written as a Rescaling layer
    # so the saved model has no Lambda layer and reloads cleanly in the Flask app.
    x = L.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)                               # keep BatchNorm in inference mode
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="dropout")(x)
    outputs = L.Dense(1, activation="sigmoid", name="prob_ai")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_head")
    compile_model(model, learning_rate)
    return model, base


def unfreeze_last_block(base, first_layer="block_16_expand"):
    """Stage 2: make only the last inverted-residual block (block_16 + final conv) trainable.
    BatchNorm layers stay frozen so their statistics are not disturbed by the small dataset."""
    names = [l.name for l in base.layers]
    if first_layer not in names:
        raise ValueError(f"{first_layer!r} not found; last layers are: {names[-12:]}")
    cut = names.index(first_layer)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < cut or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def trainable_layer_names(model):
    """Names of all layers (including those inside the nested base model) that have trainable weights."""
    names = []
    for layer in model.layers:
        subs = layer.layers if isinstance(layer, tf.keras.Model) else [layer]
        names += [l.name for l in subs if l.trainable_weights]
    return names