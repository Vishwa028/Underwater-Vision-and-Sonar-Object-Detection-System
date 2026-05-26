import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small marine image classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root folder")
    parser.add_argument("--output", type=Path, default=Path("models") / "marine_classifier.keras")
    parser.add_argument("--labels-output", type=Path, default=Path("models") / "marine_labels.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> int:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is not installed. Install it with `pip install -r requirements-ml.txt` first."
        ) from error

    parser = build_arg_parser()
    args = parser.parse_args()

    train_dir = args.dataset / "train"
    val_dir = args.dataset / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Dataset folders `dataset/train` and `dataset/val` are required.")

    image_size = (args.image_size, args.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="marine_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="marine_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.4, monitor="val_loss"),
    ]

    model.fit(train_ds.prefetch(tf.data.AUTOTUNE), validation_data=val_ds.prefetch(tf.data.AUTOTUNE), epochs=args.epochs, callbacks=callbacks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    args.labels_output.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print(f"Saved model to {args.output}")
    print(f"Saved labels to {args.labels_output}")
    print("Classes:", ", ".join(class_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
