"""
Sequence-level preprocessing shared by train_model.py and test_model.py.

Key responsibilities:
  - resample a variable-length sequence of frame feature vectors to a
    fixed length (MODEL_SEQUENCE_LEN) without discarding temporal shape
  - fit/apply a StandardScaler at the *frame* level (fit only on frames
    that belong to the training sequences, to avoid leakage)
  - split whole sequences (never individual frames) into train/val/test
"""

import numpy as np
from . import config


def resample_sequence(frames, target_len=config.MODEL_SEQUENCE_LEN):
    """
    Resizes a (T, F) array of per-frame features to (target_len, F).

    - If T > target_len: uniformly sample target_len frames (keeps the
      overall motion shape rather than just clipping the end).
    - If T < target_len: pad by repeating the last frame.
    - If T == target_len: unchanged.
    """
    frames = np.asarray(frames, dtype=np.float32)
    t = frames.shape[0]

    if t == target_len:
        return frames

    if t > target_len:
        indices = np.linspace(0, t - 1, target_len).astype(int)
        return frames[indices]

    # t < target_len: pad with the last frame
    pad_count = target_len - t
    pad = np.repeat(frames[-1:], pad_count, axis=0)
    return np.concatenate([frames, pad], axis=0)


def sequences_to_array(sequences):
    """List of (T_i, F) arrays -> single (N, MODEL_SEQUENCE_LEN, F) array."""
    resampled = [resample_sequence(seq) for seq in sequences]
    return np.stack(resampled, axis=0)


def split_sequences(sequence_ids, labels, test_size=0.15, val_size=0.15,
                     random_state=config.RANDOM_SEED):
    from sklearn.model_selection import train_test_split
    """
    Splits at the SEQUENCE level (never frame level) to avoid leakage:
    all frames of a given sequence_id always end up in exactly one of
    train/val/test.

    sequence_ids: list/array of unique sequence identifiers, one per sequence
    labels: list/array of the label for each sequence (same order)

    Returns three lists of sequence_ids: train_ids, val_ids, test_ids
    """
    ids = np.array(sequence_ids)
    labels = np.array(labels)

    # Stratify when every class has enough sequences; fall back to a
    # plain random split for tiny classes so this never crashes.
    try:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            ids, labels, test_size=(test_size + val_size),
            random_state=random_state, stratify=labels,
        )
        relative_test = test_size / (test_size + val_size)
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=relative_test,
            random_state=random_state, stratify=temp_y,
        )
    except ValueError:
        train_ids, temp_ids = train_test_split(
            ids, test_size=(test_size + val_size), random_state=random_state,
        )
        relative_test = test_size / (test_size + val_size)
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=relative_test, random_state=random_state,
        )

    return list(train_ids), list(val_ids), list(test_ids)


def fit_scaler_on_frames(scaler, sequences_train):
    """
    Fits a sklearn scaler on every individual frame from the TRAINING
    sequences only (flattened over time), then returns the fitted scaler.
    """
    all_frames = np.concatenate([np.asarray(s, dtype=np.float32) for s in sequences_train], axis=0)
    scaler.fit(all_frames)
    return scaler


def apply_scaler_to_sequences(scaler, sequences):
    """Applies an already-fitted scaler to every frame of every sequence."""
    scaled = []
    for seq in sequences:
        seq = np.asarray(seq, dtype=np.float32)
        shape = seq.shape
        flat = seq.reshape(-1, shape[-1])
        flat_scaled = scaler.transform(flat)
        scaled.append(flat_scaled.reshape(shape))
    return scaled
