"""
test_model.py
==============
Real-time sign-language recognition. Run with no arguments:

    python test_model.py

It immediately:
    1. Loads the trained model + label encoder + scaler + config.json
       produced by train_model.py.
    2. Opens the webcam.
    3. Detects both hands and the face every frame using exactly the same
       feature pipeline used during training (utils/features.py).
    4. Maintains a rolling buffer of recent frames, resamples it to the
       model's fixed sequence length, and runs the model on it.
    5. Smooths predictions over time (voting + confidence threshold) so
       the displayed sign doesn't flicker frame to frame.
    6. Builds recognized words into a subtitle sentence (max 12 words per
       sentence - the 13th word starts a fresh sentence).
    7. Applies filler-word rules (FILLER_RULES) to the subtitle only, e.g.
       MY NAME -> MY NAME IS. Fillers are never model predictions.
"""

import collections
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

from utils import config
from utils.features import build_feature_vector, FEATURE_NAMES
from utils.landmarks import LandmarkDetector, draw_landmarks, hand_bounding_box
from utils.preprocessing import resample_sequence
from utils.inference import load_inference_pickle


class SignModel:
    """Wraps either a Keras LSTM model or a scikit-learn RandomForest
    behind one common `predict_proba(sequence)` interface."""

    def __init__(self, run_config):
        self.run_config = run_config
        self.model_type = run_config["model_type"]

        if self.model_type == "lstm":
            import tensorflow as tf
            from tensorflow import keras
            self.model = keras.models.load_model(run_config["model_path"], compile=False)
            self._infer = tf.function(
                lambda batch: self.model(batch, training=False),
                reduce_retracing=True,
            )
        else:
            with open(run_config["model_path"], "rb") as f:
                self.model = pickle.load(f)

    def predict_proba(self, fixed_sequence):
        """fixed_sequence: (sequence_len, num_features) scaled array."""
        if self.model_type == "lstm":
            batch = fixed_sequence[np.newaxis, ...]
            # Reuse the traced graph for every live frame.
            probs = self._infer(batch).numpy()[0]
            return probs
        else:
            mean = fixed_sequence.mean(axis=0)
            std = fixed_sequence.std(axis=0)
            mn = fixed_sequence.min(axis=0)
            mx = fixed_sequence.max(axis=0)
            agg = np.concatenate([mean, std, mn, mx])[np.newaxis, :]
            probs = self.model.predict_proba(agg)[0]
            return probs


def load_artifacts():
    for path in (config.CONFIG_FILE, config.LABEL_ENCODER_FILE, config.SCALER_FILE):
        if not os.path.exists(path):
            print(f"Missing {path}. Run train_model.py first.")
            sys.exit(1)

    with open(config.CONFIG_FILE) as f:
        run_config = json.load(f)
    run_config["model_path"] = (
        config.MODEL_FILE_KERAS if run_config["model_type"] == "lstm"
        else config.MODEL_FILE_SKLEARN
    )

    label_encoder = load_inference_pickle(config.LABEL_ENCODER_FILE)
    scaler = load_inference_pickle(config.SCALER_FILE)

    if run_config["feature_length"] != len(FEATURE_NAMES):
        print(
            "WARNING: the loaded model was trained with a different feature "
            "vector length than the current utils/features.py. Predictions "
            "will likely be wrong. Retrain the model."
        )

    model = SignModel(run_config)
    return model, label_encoder, scaler, run_config


def split_class_name(class_name, separator):
    if separator in class_name:
        language, label = class_name.split(separator, 1)
        return language, label
    return "", class_name


# --- Filler-word rules (subtitle layer only, never a model prediction) ---
# Key = consecutive recognized signs, value = word inserted right after them.
# To add more later, just add entries, e.g. ("HOW", "YOU"): "ARE"
FILLER_RULES = {
    ("MY", "NAME"): "IS",
}


def apply_fillers(words):
    """Takes the raw recognized signs and returns the display words with
    filler words inserted. The raw list is never modified."""
    out = []
    n = len(words)
    for i, word in enumerate(words):
        out.append(word)
        if i == 0:
            continue
        filler = FILLER_RULES.get((words[i - 1].upper(), word.upper()))
        if filler is None:
            continue
        # If the model really recognized the filler as the next sign, skip
        # inserting it so we don't get "IS IS".
        next_is_filler = i + 1 < n and words[i + 1].upper() == filler
        if not next_is_filler:
            out.append(filler)
    return out


def draw_prediction_panel(frame, language, label, confidence):
    text_sign = f"SIGN: {label if label else 'UNKNOWN'}"
    text_conf = f"CONFIDENCE: {confidence * 100:.1f}%" if label else "CONFIDENCE: --"
    cv2.rectangle(frame, (0, 0), (330, 70), (0, 0, 0), -1)
    cv2.putText(frame, text_sign, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if label else (0, 0, 255), 2)
    cv2.putText(frame, text_conf, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0) if label else (0, 0, 255), 2)
    if language:
        cv2.putText(frame, f"Language: {language}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_subtitle(frame, sentence, image_w, image_h):
    """Draws the current sentence-in-progress as a subtitle bar at the bottom."""
    if not sentence:
        return
    bar_height = 50
    y = image_h - bar_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y), (image_w, image_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, sentence, (10, image_h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)


def draw_hand_prediction_boxes(frame, detection, image_w, image_h, label, confidence):
    for hand_name, color in (("Left", (255, 0, 0)), ("Right", (0, 255, 0))):
        pts = detection["hands"][hand_name]
        if pts is not None:
            x1, y1, x2, y2 = hand_bounding_box(pts, image_w, image_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, f"{label} {confidence * 100:.0f}%",
                            (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)


def main():
    model, label_encoder, scaler, run_config = load_artifacts()
    sequence_len = run_config["sequence_len"]
    separator = run_config.get("class_separator", "::")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = LandmarkDetector()
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    frame_buffer = collections.deque(maxlen=config.MAX_SEQUENCE_LEN)
    prediction_buffer = collections.deque(maxlen=config.PREDICTION_BUFFER_SIZE)

    min_frame_interval = 1.0 / config.TARGET_CAPTURE_FPS
    last_capture = time.time()

    displayed_language, displayed_label, displayed_conf = "", "", 0.0

    # --- Sentence/subtitle state ---
    sentence_words = []
    last_added_word = None
    MAX_SENTENCE_WORDS = 12

    # --- Buffer clearing state ---
    no_hands_start = None
    NO_HANDS_CLEAR_DELAY = 0.5  # seconds without hands before clearing buffer

    print("Camera opened. Show a sign to the camera.")
    print(f"Classes known to the model: {list(label_encoder.classes_)}\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)
            detection = detector.process(frame)

            hands_present = (detection["hands"]["Left"] is not None or
                             detection["hands"]["Right"] is not None)

            now = time.time()
            if now - last_capture >= min_frame_interval:
                vec = build_feature_vector(detection, image_w, image_h)
                # Only buffer frames when at least one hand is detected
                if hands_present:
                    frame_buffer.append(vec)
                last_capture = now

            # Clear buffers when hands disappear for a while
            if not hands_present:
                if no_hands_start is None:
                    no_hands_start = time.time()
                elif time.time() - no_hands_start > NO_HANDS_CLEAR_DELAY:
                    frame_buffer.clear()
                    prediction_buffer.clear()
                    displayed_language, displayed_label, displayed_conf = "", "", 0.0
                    no_hands_start = None
            else:
                no_hands_start = None

            if len(frame_buffer) >= config.MIN_SEQUENCE_LEN:
                raw_seq = np.stack(list(frame_buffer), axis=0)
                fixed_seq = resample_sequence(raw_seq, target_len=sequence_len)
                scaled_seq = scaler.transform(fixed_seq)

                probs = model.predict_proba(scaled_seq)
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                top_class = label_encoder.classes_[top_idx]

                if top_conf >= config.MIN_CONFIDENCE_THRESHOLD:
                    prediction_buffer.append((top_class, top_conf))
                else:
                    prediction_buffer.append((None, top_conf))

                # Majority vote over the recent prediction buffer.
                votes = collections.Counter(
                    c for c, _ in prediction_buffer if c is not None
                )
                if votes:
                    best_class, count = votes.most_common(1)[0]
                    if count >= config.MIN_CONSECUTIVE_AGREEMENT:
                        confs = [c for cls, c in prediction_buffer if cls == best_class]
                        lang, label = split_class_name(best_class, separator)
                        if label.startswith("NOT_"):
                            # A "NOT_x" class winning just means "no recognized sign" -
                            # never surface it as a label, treat it like UNKNOWN instead.
                            displayed_language, displayed_label, displayed_conf = "", "", 0.0
                        else:
                            displayed_language, displayed_label = lang, label
                            displayed_conf = float(np.mean(confs))
                    else:
                        displayed_language, displayed_label, displayed_conf = "", "", 0.0
                else:
                    displayed_language, displayed_label, displayed_conf = "", "", 0.0

                # --- Sentence-building logic ---
                is_real_word = displayed_label and not displayed_label.startswith("NOT_")

                if is_real_word and displayed_label != last_added_word:
                    if len(sentence_words) >= MAX_SENTENCE_WORDS:
                        sentence_words = [displayed_label]   # 13th word -> start fresh sentence
                    else:
                        sentence_words.append(displayed_label)
                    last_added_word = displayed_label
                elif not is_real_word:
                    last_added_word = None   # sign released -> same word can be added again next time

            draw_landmarks(frame, detection)
            draw_hand_prediction_boxes(frame, detection, image_w, image_h,
                                        displayed_label, displayed_conf)
            draw_prediction_panel(frame, displayed_language, displayed_label,
                                   displayed_conf)
            draw_subtitle(frame, " ".join(apply_fillers(sentence_words)), image_w, image_h)

            cv2.imshow("Sign Language Recognition", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
