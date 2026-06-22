"""
train_knocknet.py — KnockNet-lite 聲學模型訓練
基於 YAMNet / 自定義 CNN 的榴槤敲擊音分類

Phase 1: 自定義 CNN on Mel Spectrogram (輕量、快速)
Phase 3: KnockNet 雙流 ConvMixer (MFCC + MGDCC)

訓練完成後導出 TFLite 模型供瀏覽器端推理
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Phase 1: Custom CNN on Mel Spectrogram
# ============================================================

def build_cnn_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    Lightweight CNN for mel spectrogram classification.
    Input: (n_mels, time_frames, 1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Classification head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    return model


# ============================================================
# Phase 1 Alt: YAMNet Transfer Learning
# ============================================================

def build_yamnet_model(num_classes: int) -> keras.Model:
    """
    YAMNet-based transfer learning model.
    YAMNet outputs 1024-dim embeddings from 16kHz mono audio.
    """
    # Load YAMNet as feature extractor (frozen)
    yamnet_model = keras.models.load_model(
        "https://tfhub.dev/google/yamnet/1"
    )

    inputs = keras.Input(shape=(16000,), dtype=tf.float32)
    # YAMNet expects raw waveform, outputs (batch, 1024) embeddings
    embeddings = yamnet_model(inputs)

    x = layers.Dense(256, activation='relu')(embeddings)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Freeze YAMNet layers
    for layer in yamnet_model.layers:
        layer.trainable = False

    return model


# ============================================================
# Phase 3: KnockNet Dual-Stream ConvMixer (Future)
# ============================================================

def build_knocknet_dual_stream(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    KnockNet: Dual-stream ConvMixer with LCA.
    Stream 1: MFCC (magnitude domain)
    Stream 2: MGDCC (phase domain)
    Fused via Local Channel Attention + ConvMixer blocks.
    """
    # Magnitude stream
    mag_input = layers.Input(shape=input_shape, name='magnitude')
    x1 = layers.Conv2D(32, (3, 3), padding='same', activation='gelu')(mag_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = _convmixer_block(x1, 32)
    x1 = _convmixer_block(x1, 32)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Phase stream (placeholder - MGDCC extraction needs custom code)
    phase_input = layers.Input(shape=input_shape, name='phase')
    x2 = layers.Conv2D(32, (3, 3), padding='same', activation='gelu')(phase_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = _convmixer_block(x2, 32)
    x2 = _convmixer_block(x2, 32)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Fusion with Local Channel Attention (LCA)
    fused = layers.Concatenate()([x1, x2])
    fused = _local_channel_attention(fused, 64)

    # Classification
    x = layers.GlobalAveragePooling2D()(fused)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=[mag_input, phase_input], outputs=outputs)


def _convmixer_block(x: tf.Tensor, filters: int, kernel_size: int = 5) -> tf.Tensor:
    """ConvMixer block: Depthwise conv + Pointwise conv with residual."""
    # Depthwise conv (spatial mixing)
    residual = x
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), padding='same', activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])

    # Pointwise conv (channel mixing)
    x = layers.Conv2D(filters, (1, 1), activation='gelu')(x)
    x = layers.BatchNormalization()(x)

    return x


def _local_channel_attention(x: tf.Tensor, reduction: int = 16) -> tf.Tensor:
    """Local Channel Attention (LCA) module."""
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // reduction, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])


# ============================================================
# Training Pipeline
# ============================================================

def load_data(data_dir: str, feature_type: str = "mel"):
    """Load preprocessed data."""
    train_path = os.path.join(data_dir, f"dalvii_{feature_type}_train.npz")
    val_path = os.path.join(data_dir, f"dalvii_{feature_type}_val.npz")
    test_path = os.path.join(data_dir, f"dalvii_{feature_type}_test.npz")

    # Try with other dataset prefixes
    for prefix in ["dalvii", "zenodo", "combined"]:
        train_path = os.path.join(data_dir, f"{prefix}_{feature_type}_train.npz")
        val_path = os.path.join(data_dir, f"{prefix}_{feature_type}_val.npz")
        test_path = os.path.join(data_dir, f"{prefix}_{feature_type}_test.npz")
        if os.path.exists(train_path):
            break

    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}")
        print("Run prepare_audio.py first!")
        sys.exit(1)

    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)

    # Load class names
    classes_path = os.path.join(data_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = sorted(set(train['y']))

    # Encode labels
    label_map = {name: i for i, name in enumerate(class_names)}
    y_train = np.array([label_map[l] for l in train['y']])
    y_val = np.array([label_map[l] for l in val['y']])
    y_test = np.array([label_map[l] for l in test['y']])

    return train['X'], y_train, val['X'], y_val, test['X'], y_test, class_names


def train(
    data_dir: str,
    model_type: str = "cnn",
    feature_type: str = "mel",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: str = "ml/models/acoustic",
):
    """Train acoustic model."""
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_data(data_dir, feature_type)

    num_classes = len(class_names)
    input_shape = X_train.shape[1:]  # (H, W, 1)

    print(f"Input shape: {input_shape}")
    print(f"Classes: {class_names} ({num_classes})")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Build model
    if model_type == "cnn":
        model = build_cnn_model(input_shape, num_classes)
    elif model_type == "yamnet":
        model = build_yamnet_model(num_classes)
    elif model_type == "knocknet":
        model = build_knocknet_dual_stream(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            save_best_only=True, monitor='val_accuracy'
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save labels
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        for cls in class_names:
            f.write(cls + '\n')

    # Export TFLite
    export_tflite(model, output_dir, class_names)

    return model, history


def export_tflite(model: keras.Model, output_dir: str, class_names: list[str]):
    """Export model to TFLite with INT8 quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, 'knocknet_lite.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")

    # Also save FP16 version for comparison
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()

    fp16_path = os.path.join(output_dir, 'knocknet_lite_fp16.tflite')
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)

    print(f"TFLite FP16 model saved: {fp16_path} ({len(tflite_fp16)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train KnockNet-lite acoustic model")
    parser.add_argument("--data", default="ml/data/processed/audio", help="Processed data directory")
    parser.add_argument("--model", default="cnn", choices=["cnn", "yamnet", "knocknet"], help="Model architecture")
    parser.add_argument("--feature", default="mel", choices=["mel", "mfcc"], help="Feature type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="ml/models/acoustic", help="Output directory")

    args = parser.parse_args()
    train(args.data, args.model, args.feature, args.epochs, args.batch, args.lr, args.output)


if __name__ == "__main__":
    main()
