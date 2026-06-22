"""
train_cnn.py — 榴槤成熟度視覺 CNN 訓練
基於 MobileNetV2 遷移學習，導出 TFLite 供瀏覽器端推理

Phase 1: MobileNetV2 transfer learning (快速、高效)
Phase 3: EfficientNet-Lite0 (更高精度)
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet_model(num_classes: int, img_size: int = 224) -> keras.Model:
    """
    MobileNetV2 transfer learning model.
    Freezes base, adds custom classification head.
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  # Freeze base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


def build_efficientnet_model(num_classes: int, img_size: int = 224) -> keras.Model:
    """
    EfficientNetV2-S transfer learning model (Phase 3).
    """
    base = keras.applications.EfficientNetV2S(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)


def build_simple_cnn(num_classes: int, img_size: int = 224) -> keras.Model:
    """
    Lightweight custom CNN (for comparison / when MobileNet is too heavy).
    """
    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def load_data(data_dir: str):
    """Load preprocessed vision data."""
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f"vision_{split}.npz")
        if not os.path.exists(path):
            print(f"Data not found: {path}")
            print("Run prepare_vision.py first!")
            sys.exit(1)

    train = np.load(os.path.join(data_dir, "vision_train.npz"))
    val = np.load(os.path.join(data_dir, "vision_val.npz"))
    test = np.load(os.path.join(data_dir, "vision_test.npz"))

    classes_path = os.path.join(data_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = sorted(set(train['y']))

    label_map = {name: i for i, name in enumerate(class_names)}
    y_train = np.array([label_map[l] for l in train['y']])
    y_val = np.array([label_map[l] for l in val['y']])
    y_test = np.array([label_map[l] for l in test['y']])

    return train['X'], y_train, val['X'], y_val, test['X'], y_test, class_names


def train(
    data_dir: str,
    model_type: str = "mobilenet",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    fine_tune_epochs: int = 20,
    output_dir: str = "ml/models/vision",
):
    """Train vision model with optional fine-tuning."""
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data(data_dir)

    num_classes = len(class_names)
    print(f"Classes: {class_names} ({num_classes})")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")

    # Build model
    if model_type == "mobilenet":
        model = build_mobilenet_model(num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes)
    elif model_type == "simple":
        model = build_simple_cnn(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.summary()

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # Stage 1: Train classification head only
    # ==========================================
    print("\n=== Stage 1: Training classification head ===")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_1 = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model_head.keras'),
            save_best_only=True, monitor='val_accuracy'
        ),
    ]

    history_1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_1,
    )

    # ==========================================
    # Stage 2: Fine-tune (unfreeze top layers)
    # ==========================================
    if model_type in ["mobilenet", "efficientnet"] and fine_tune_epochs > 0:
        print("\n=== Stage 2: Fine-tuning top layers ===")

        # Unfreeze top 30% of base model
        base_model = model.layers[1]  # The base model
        base_model.trainable = True
        total_layers = len(base_model.layers)
        freeze_until = int(total_layers * 0.7)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        # Recompile with lower LR
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks_2 = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, 'best_model.keras'),
                save_best_only=True, monitor='val_accuracy'
            ),
        ]

        history_2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=fine_tune_epochs,
            batch_size=batch_size,
            callbacks=callbacks_2,
        )
    else:
        # Just copy the head-only model
        import shutil
        src = os.path.join(output_dir, 'best_model_head.keras')
        dst = os.path.join(output_dir, 'best_model.keras')
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save labels
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for cls in class_names:
            f.write(cls + '\n')

    # Export TFLite
    export_tflite(model, output_dir)

    return model


def export_tflite(model: keras.Model, output_dir: str):
    """Export to TFLite with INT8 quantization."""
    # INT8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    tflite_path = os.path.join(output_dir, 'durian_cnn.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite INT8: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # FP16
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    fp16_path = os.path.join(output_dir, 'durian_cnn_fp16.tflite')
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)
    print(f"TFLite FP16: {fp16_path} ({len(tflite_fp16)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train durian vision CNN")
    parser.add_argument("--data", default="ml/data/processed/vision", help="Processed data directory")
    parser.add_argument("--model", default="mobilenet", choices=["mobilenet", "efficientnet", "simple"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (head training)")
    parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="ml/models/vision", help="Output directory")

    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.batch, args.lr, args.fine_tune_epochs, args.output)


if __name__ == "__main__":
    main()
