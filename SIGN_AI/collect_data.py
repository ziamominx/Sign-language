"""
collect_data.py
================
Interactive data-collection tool for the sign-language recognition system.

Usage:
    python collect_data.py

Flow:
    1. Asks which sign language you're collecting (e.g. ASL, ISL, BSL).
    2. Asks which sign label you want to record (e.g. NO, FATHER).
    3. Opens the webcam with live hand + face landmark overlay.
    4. Press 1 to record a POSITIVE example of the current label
       (~5 seconds of movement, captured as a temporal sequence).
       Press 2 to record a NEGATIVE example (auto-labelled NOT_<label>).
       Press N to switch to a new sign label without restarting the camera.
       Press Q to quit.
    5. Every recorded sequence is appended to dataset/dataset.csv, one row
       per (sequence, frame) so the full temporal sequence can be rebuilt
       later. A summary row is also appended to dataset/dataset_metadata.csv.

The program never overwrites existing data -- it only appends.
"""

import csv
import os
import time
import uuid

import cv2

from utils import config
from utils.features import build_feature_vector, FEATURE_NAMES
from utils.landmarks import LandmarkDetector, draw_landmarks, hand_bounding_box

CSV_HEADER = ["language", "label", "sequence_id", "frame"] + FEATURE_NAMES
METADATA_HEADER = ["language", "label", "sequence_id", "num_frames", "timestamp"]


def ensure_csv_headers():
    if not os.path.exists(config.DATASET_CSV):
        with open(config.DATASET_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
    if not os.path.exists(config.DATASET_METADATA_CSV):
        with open(config.DATASET_METADATA_CSV, "w", newline="") as f:
            csv.writer(f).writerow(METADATA_HEADER)


def append_sequence_to_csv(language, label, sequence_id, frame_vectors):
    with open(config.DATASET_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for frame_idx, vec in enumerate(frame_vectors):
            writer.writerow([language, label, sequence_id, frame_idx] + vec.tolist())

    with open(config.DATASET_METADATA_CSV, "a", newline="") as f:
        csv.writer(f).writerow(
            [language, label, sequence_id, len(frame_vectors), time.time()]
        )


def draw_overlay(frame, language, label, status_lines, recording, rec_time, rec_frames):
    h, w = frame.shape[:2]
    y = 25
    cv2.putText(frame, f"Language: {language}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    y += 25
    cv2.putText(frame, f"Label: {label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    y += 30
    for line in status_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y += 20

    if recording:
        cv2.putText(frame, "RECORDING...", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Time: {rec_time:.1f} / {config.RECORDING_DURATION_SECONDS:.1f} sec   "
            f"Frames: {rec_frames}",
            (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )
        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)


def draw_hand_boxes(frame, detection, image_w, image_h):
    for hand_name, color in (("Left", (255, 0, 0)), ("Right", (0, 255, 0))):
        pts = detection["hands"][hand_name]
        if pts is not None:
            x1, y1, x2, y2 = hand_bounding_box(pts, image_w, image_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, hand_name, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def record_sequence(cap, detector, language, label, image_w, image_h):
    """
    Records ~config.RECORDING_DURATION_SECONDS of frames.
    Returns a list of per-frame feature vectors (np.array), or None if
    the recording was rejected (e.g. camera failure, too few frames).
    """
    frame_vectors = []
    start = time.time()
    min_frame_interval = 1.0 / config.TARGET_CAPTURE_FPS
    last_capture = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed during recording.")
            break
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start
        detection = detector.process(frame)

        # Throttle capture to roughly TARGET_CAPTURE_FPS regardless of the
        # webcam's actual frame rate, so sequence length stays comparable
        # across machines.
        if elapsed - last_capture >= min_frame_interval:
            vec = build_feature_vector(detection, image_w, image_h)
            frame_vectors.append(vec)
            last_capture = elapsed

        draw_landmarks(frame, detection)
        draw_hand_boxes(frame, detection, image_w, image_h)
        draw_overlay(
            frame, language, label,
            status_lines=["Press S to stop early"],
            recording=True, rec_time=elapsed, rec_frames=len(frame_vectors),
        )
        cv2.imshow("Sign Language Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") or elapsed >= config.RECORDING_DURATION_SECONDS:
            break
        if key == ord("q"):
            return None  # abort whole program from mid-recording

        if len(frame_vectors) >= config.MAX_SEQUENCE_LEN:
            break

    if len(frame_vectors) < config.MIN_SEQUENCE_LEN:
        print(
            f"Sequence rejected: only {len(frame_vectors)} frames captured "
            f"(minimum {config.MIN_SEQUENCE_LEN}). Try recording again with "
            f"visible hands."
        )
        return None

    return frame_vectors


def main():
    ensure_csv_headers()

    language = input("Enter the sign language you are collecting: ").strip()
    if not language:
        print("Sign language cannot be empty.")
        return

    label = input("Enter sign label: ").strip().upper()
    if not label:
        print("Sign label cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = LandmarkDetector()
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print(f"\nCamera opened. Feature vector length: {len(FEATURE_NAMES)}\n")
    print(f"1 = Record {label}")
    print(f"2 = Record NOT_{label}")
    print("N = New label   |   S = Stop current recording early   |   Q = Quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)
            detection = detector.process(frame)

            draw_landmarks(frame, detection)
            draw_hand_boxes(frame, detection, image_w, image_h)
            draw_overlay(
                frame, language, label,
                status_lines=[
                    f"1 = Record {label}   2 = Record NOT_{label}",
                    "N = New label   Q = Quit",
                ],
                recording=False, rec_time=0, rec_frames=0,
            )
            cv2.imshow("Sign Language Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("n"):
                new_label = input("Enter sign label: ").strip().upper()
                if new_label:
                    label = new_label
                    print(f"\nNow collecting for label: {label}")
                    print(f"1 = Record {label}")
                    print(f"2 = Record NOT_{label}\n")

            elif key in (ord("1"), ord("2")):
                target_label = label if key == ord("1") else f"NOT_{label}"
                print(f"Recording '{target_label}' ...")
                frames = record_sequence(cap, detector, language, target_label,
                                          image_w, image_h)
                if frames is None:
                    print("(no camera abort or rejected sequence)")
                    continue
                sequence_id = uuid.uuid4().hex[:12]
                append_sequence_to_csv(language, target_label, sequence_id, frames)
                print(
                    f"Sequence recorded successfully.\n"
                    f"  Label: {target_label}\n"
                    f"  Frames: {len(frames)}\n"
                    f"  Language: {language}\n"
                    f"  Sequence ID: {sequence_id}\n"
                    f"  Saved to {config.DATASET_CSV}\n"
                )

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Data collection session ended.")


if __name__ == "__main__":
    main()
