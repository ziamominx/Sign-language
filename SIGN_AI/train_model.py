"""
train_model.py
===============
Fully automatic training program. Run with no arguments:

    python train_model.py

It will:
    1. Load dataset/dataset.csv.
    2. Detect all available (language, label) classes automatically.
    3. Group rows by sequence_id and reconstruct each temporal sequence
       in frame order.
    4. Split at the SEQUENCE level into train/val/test (no leakage).
    5. Resample every sequence to a fixed length and scale features
       (scaler fit on training frames only).
    6. Train a temporal (LSTM) model if TensorFlow is available, else
       fall back to a RandomForest on aggregated per-sequence statistics.
    7. Evaluate on the held-out validation and test sets.
    8. Save the model + label encoder + scaler + config.json to models/.

New sequences added to the CSV are picked up automatically next time
this script is run -- nothing needs to change in this file.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from utils import config
from utils.features import FEATURE_NAMES
from utils.preprocessing import (
    resample_sequence, fit_scaler_on_frames, apply_scaler_to_sequences,
    split_sequences,
)

CLASS_SEPARATOR = "::"  # joins language + label into one training class


def load_dataset():
    if not os.path.exists(config.DATASET_CSV):
        print(f"No dataset found at {config.DATASET_CSV}. Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(config.DATASET_CSV)
    if df.empty:
        print("Dataset is empty. Collect some data first.")
        sys.exit(1)
    return df


def reconstruct_sequences(df):
    """
    Groups rows by sequence_id, sorts by frame, and returns:
        sequences: list of (T_i, F) np.float32 arrays
        class_names: list of str, one per sequence ("LANGUAGE::LABEL")
        sequence_ids: list of sequence_id strings
    """
    feature_cols = FEATURE_NAMES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(
            "Dataset columns do not match the current feature pipeline "
            f"(missing {len(missing)} columns, e.g. {missing[:3]}). "
            "The dataset may have been collected with a different version "
            "of utils/features.py. Re-collect data or update the pipeline."
        )
        sys.exit(1)

    sequences, class_names, sequence_ids = [], [], []
    for seq_id, group in df.groupby("sequence_id"):
        group = group.sort_values("frame")
        seq_array = group[feature_cols].to_numpy(dtype=np.float32)
        language = group["language"].iloc[0]
        label = group["label"].iloc[0]
        sequences.append(seq_array)
        class_names.append(f"{language}{CLASS_SEPARATOR}{label}")
        sequence_ids.append(seq_id)

    return sequences, class_names, sequence_ids


def build_lstm_model(input_shape, num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def aggregate_sequence_features(sequences_array):
    """
    Fallback feature representation for the sklearn path: per-sequence
    mean/std/min/max across time, turning (N, T, F) into (N, 4*F).
    Loses fine-grained temporal order but keeps overall motion statistics.
    """
    mean = sequences_array.mean(axis=1)
    std = sequences_array.std(axis=1)
    mn = sequences_array.min(axis=1)
    mx = sequences_array.max(axis=1)
    return np.concatenate([mean, std, mn, mx], axis=1)


def main():
    print("Loading dataset...")
    df = load_dataset()

    sequences, class_names, sequence_ids = reconstruct_sequences(df)
    languages = sorted(set(c.split(CLASS_SEPARATOR)[0] for c in class_names))
    unique_classes = sorted(set(class_names))

    print(f"Languages found: {', '.join(languages)}")
    print("Classes found:")
    for c in unique_classes:
        print(f"  {c}")
    print(f"\nSequences: {len(sequences)}")
    print(f"Frames: {sum(len(s) for s in sequences)}\n")

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(class_names)

    train_ids, val_ids, test_ids = split_sequences(sequence_ids, class_names)
    id_to_index = {sid: i for i, sid in enumerate(sequence_ids)}

    def subset(ids):
        idx = [id_to_index[i] for i in ids]
        return [sequences[i] for i in idx], y_all[idx]

    seq_train, y_train = subset(train_ids)
    seq_val, y_val = subset(val_ids)
    seq_test, y_test = subset(test_ids)

    print(f"Training sequences: {len(seq_train)}")
    print(f"Validation sequences: {len(seq_val)}")
    print(f"Test sequences: {len(seq_test)}")
    print(f"Classes: {len(unique_classes)}\n")

    # --- Data augmentation (training only) ---
    def augment_sequence(seq):
        """Generate augmented copies of a sequence."""
        augmented = [seq]
        T = len(seq)
        # 1. Time-stretch: randomly speed up/slow down
        for _ in range(2):
            stretch = np.random.uniform(0.8, 1.2)
            new_T = max(config.MIN_SEQUENCE_LEN, int(T * stretch))
            idx = np.linspace(0, T - 1, new_T).astype(int)
            augmented.append(seq[idx])
        # 2. Gaussian noise on features (small)
        noise = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)
        augmented.append(noise)
        return augmented

    augmented_train, augmented_y = [], []
    for s, y in zip(seq_train, y_train):
        for aug in augment_sequence(s):
            augmented_train.append(aug)
            augmented_y.append(y)
    seq_train = augmented_train
    y_train = np.array(augmented_y)
    print(f"After augmentation: {len(seq_train)} training sequences (4x)\n")

    # Resample every sequence to a fixed length BEFORE fitting the scaler,
    # so the scaler sees the same frame distribution used at inference.
    train_fixed = [resample_sequence(s) for s in seq_train]
    val_fixed = [resample_sequence(s) for s in seq_val]
    test_fixed = [resample_sequence(s) for s in seq_test]

    scaler = StandardScaler()
    fit_scaler_on_frames(scaler, train_fixed)

    train_scaled = apply_scaler_to_sequences(scaler, train_fixed)
    val_scaled = apply_scaler_to_sequences(scaler, val_fixed)
    test_scaled = apply_scaler_to_sequences(scaler, test_fixed)

    X_train = np.stack(train_scaled, axis=0)
    X_val = np.stack(val_scaled, axis=0) if val_scaled else None
    X_test = np.stack(test_scaled, axis=0) if test_scaled else None

    print("Training model...")

    model_type = "lstm"
    try:
        import tensorflow as tf  # noqa: F401
        model = build_lstm_model(
            input_shape=(config.MODEL_SEQUENCE_LEN, len(FEATURE_NAMES)),
            num_classes=len(unique_classes),
        )
        callbacks = []
        try:
            from tensorflow import keras
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                )
            )
        except Exception:
            pass

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and len(X_val) else None,
            epochs=60, batch_size=8, verbose=2, callbacks=callbacks,
        )

        if X_val is not None and len(X_val):
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        else:
            val_acc = history.history.get("accuracy", [0])[-1]

        if X_test is not None and len(X_test):
            test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            test_acc = accuracy_score(y_test, test_pred)
            print("\nTest set report:")
            print(classification_report(
                y_test, test_pred,
                labels=list(range(len(label_encoder.classes_))),
                target_names=label_encoder.classes_, zero_division=0,
            ))
        else:
            test_acc = None

        model.save(config.MODEL_FILE_KERAS)
        model_path_used = config.MODEL_FILE_KERAS

    except ImportError:
        print("TensorFlow not available -- falling back to a RandomForest "
              "classifier on aggregated per-sequence statistics.\n")
        model_type = "random_forest"
        from sklearn.ensemble import RandomForestClassifier

        X_train_agg = aggregate_sequence_features(X_train)
        model = RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=-1
        )
        model.fit(X_train_agg, y_train)

        if X_val is not None and len(X_val):
            X_val_agg = aggregate_sequence_features(X_val)
            val_acc = accuracy_score(y_val, model.predict(X_val_agg))
        else:
            val_acc = accuracy_score(y_train, model.predict(X_train_agg))

        if X_test is not None and len(X_test):
            X_test_agg = aggregate_sequence_features(X_test)
            test_pred = model.predict(X_test_agg)
            test_acc = accuracy_score(y_test, test_pred)
            print("\nTest set report:")
            print(classification_report(
                y_test, test_pred,
                labels=list(range(len(label_encoder.classes_))),
                target_names=label_encoder.classes_, zero_division=0,
            ))
        else:
            test_acc = None

        with open(config.MODEL_FILE_SKLEARN, "wb") as f:
            pickle.dump(model, f)
        model_path_used = config.MODEL_FILE_SKLEARN

    print(f"\nValidation accuracy: {val_acc * 100:.2f}%")
    if test_acc is not None:
        print(f"Test accuracy: {test_acc * 100:.2f}%")

    print("\nSaving model...")
    with open(config.LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    run_config = {
        "model_type": model_type,
        "model_path": model_path_used,
        "sequence_len": config.MODEL_SEQUENCE_LEN,
        "feature_length": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "classes": list(label_encoder.classes_),
        "languages": languages,
        "class_separator": CLASS_SEPARATOR,
        "num_train_sequences": len(seq_train),
        "num_val_sequences": len(seq_val),
        "num_test_sequences": len(seq_test),
        "validation_accuracy": float(val_acc),
        "test_accuracy": float(test_acc) if test_acc is not None else None,
    }
    with open(config.CONFIG_FILE, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Model saved to:\n  {model_path_used}")
    print(f"Also saved: {config.LABEL_ENCODER_FILE}, {config.SCALER_FILE}, "
          f"{config.CONFIG_FILE}")


if __name__ == "__main__":
    main()
