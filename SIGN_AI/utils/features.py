"""
Turns one frame of raw MediaPipe landmarks (from landmarks.py) into a
single flat, fixed-length numeric feature vector.

This module is the ONE place that defines "what a frame looks like as
numbers". collect_data.py, train_model.py and test_model.py all import
`build_feature_vector` / `FEATURE_NAMES` from here, so the feature
representation can never drift between collection and inference.

Feature groups (all normalized, see the two `_normalize_*` functions):
  1. Left hand landmarks   (21 * 3 = 63)   -- zeros if hand absent
  2. Right hand landmarks  (21 * 3 = 63)   -- zeros if hand absent
  3. Hand presence flags   (2)             -- 1.0 if detected else 0.0
  4. Face landmarks        (len(FACE_LANDMARK_NAMES) * 3)
  5. Face presence flag    (1)
  6. Hand-to-face distances (for each hand x each hand-reference point x
     each face-reference point) -- captures things like "thumb near
     forehead" without ever hard-coding what that means.
  7. Hand position relative to face center (2 hands * 2 (x,y))

Everything is derived purely from geometry; nothing here encodes which
sign means what -- that is left entirely to the trained model.
"""

import numpy as np

from . import config

HAND_NAMES = ["Left", "Right"]


def _safe_get(points, idx):
    return np.array(points[idx], dtype=np.float32) if points is not None else None


def _normalize_hand(hand_points):
    """
    Centers a hand's 21 landmarks on the wrist and scales by the
    wrist-to-middle-finger-MCP distance, so the representation is
    invariant to hand size and camera distance.

    Returns an (21, 3) float32 array, or None if hand_points is None.
    """
    if hand_points is None:
        return None
    pts = np.array(hand_points, dtype=np.float32)  # (21, 3)
    wrist = pts[config.HAND_WRIST_IDX]
    scale_ref = pts[config.HAND_MIDDLE_MCP_IDX]
    scale = np.linalg.norm(scale_ref[:2] - wrist[:2])
    scale = scale if scale > 1e-6 else 1e-6
    normalized = (pts - wrist) / scale
    return normalized


def _normalize_face(face_points):
    """
    Centers face landmarks on the nose tip and scales by inter-eye
    distance, so the representation is invariant to face size / camera
    distance.

    Returns a dict {name: (x, y, z)} for the landmarks in
    FACE_LANDMARK_IDS, or None if face_points is None.
    """
    if face_points is None:
        return None
    pts = np.array(face_points, dtype=np.float32)  # (478, 3) — 468 mesh + 10 iris pts
    nose = pts[config.FACE_LANDMARK_IDS["nose_tip"]]
    left_eye = pts[config.FACE_LANDMARK_IDS["left_eye_outer"]]
    right_eye = pts[config.FACE_LANDMARK_IDS["right_eye_outer"]]
    scale = np.linalg.norm(left_eye[:2] - right_eye[:2])
    scale = scale if scale > 1e-6 else 1e-6

    named = {}
    for name, idx in config.FACE_LANDMARK_IDS.items():
        named[name] = (pts[idx] - nose) / scale
    return named


def _hand_to_face_distances(raw_hand_points, raw_face_points, image_w, image_h):
    """
    Real-world-ish (pixel-space, then face-scale-normalized) distances
    between key hand points and key face points. This is what lets the
    model learn "thumb touching forehead" vs "same hand shape, no
    contact" without any hard-coded rule.

    Returns a flat list of floats, one per (hand reference point x face
    reference point) pair, in a fixed order.
    """
    distances = []

    if raw_face_points is None:
        # No face -> cannot compute meaningful distances; fill with a
        # neutral large value.
        n = len(config.HAND_REFERENCE_POINTS) * len(config.FACE_REFERENCE_POINTS)
        return [0.0] * n

    face_pts = np.array(raw_face_points, dtype=np.float32)
    left_eye = face_pts[config.FACE_LANDMARK_IDS["left_eye_outer"]]
    right_eye = face_pts[config.FACE_LANDMARK_IDS["right_eye_outer"]]
    face_scale = np.linalg.norm(
        (left_eye[:2] * [image_w, image_h]) - (right_eye[:2] * [image_w, image_h])
    )
    face_scale = face_scale if face_scale > 1e-6 else 1e-6

    face_ref_coords = {}
    for name in config.FACE_REFERENCE_POINTS:
        if name == "mouth_center":
            upper = face_pts[config.FACE_LANDMARK_IDS["upper_lip"]]
            lower = face_pts[config.FACE_LANDMARK_IDS["lower_lip"]]
            coord = (upper + lower) / 2.0
        else:
            coord = face_pts[config.FACE_LANDMARK_IDS[name]]
        face_ref_coords[name] = coord[:2] * [image_w, image_h]

    if raw_hand_points is None:
        n = len(config.HAND_REFERENCE_POINTS) * len(config.FACE_REFERENCE_POINTS)
        return [0.0] * n

    hand_pts = np.array(raw_hand_points, dtype=np.float32)
    for hand_ref_name, hand_idx in config.HAND_REFERENCE_POINTS.items():
        hand_coord = hand_pts[hand_idx][:2] * [image_w, image_h]
        for face_ref_name in config.FACE_REFERENCE_POINTS:
            face_coord = face_ref_coords[face_ref_name]
            dist = np.linalg.norm(hand_coord - face_coord) / face_scale
            distances.append(float(dist))

    return distances


def _relative_hand_position(raw_hand_points, raw_face_points, image_w, image_h):
    """Hand wrist position relative to the face center, (dx, dy), face-scale normalized."""
    if raw_hand_points is None or raw_face_points is None:
        return [0.0, 0.0]

    hand_pts = np.array(raw_hand_points, dtype=np.float32)
    face_pts = np.array(raw_face_points, dtype=np.float32)

    wrist = hand_pts[config.HAND_WRIST_IDX][:2] * [image_w, image_h]
    face_center = face_pts[config.FACE_LANDMARK_IDS["nose_tip"]][:2] * [image_w, image_h]

    left_eye = face_pts[config.FACE_LANDMARK_IDS["left_eye_outer"]][:2] * [image_w, image_h]
    right_eye = face_pts[config.FACE_LANDMARK_IDS["right_eye_outer"]][:2] * [image_w, image_h]
    face_scale = np.linalg.norm(left_eye - right_eye)
    face_scale = face_scale if face_scale > 1e-6 else 1e-6

    delta = (wrist - face_center) / face_scale
    return [float(delta[0]), float(delta[1])]


def _build_feature_names():
    names = []
    for hand in HAND_NAMES:
        for i in range(config.NUM_HAND_LANDMARKS):
            for axis in ("x", "y", "z"):
                names.append(f"{hand.lower()}_hand_lm{i}_{axis}")
    for hand in HAND_NAMES:
        names.append(f"{hand.lower()}_hand_present")
    for face_name in config.FACE_LANDMARK_NAMES:
        for axis in ("x", "y", "z"):
            names.append(f"face_{face_name}_{axis}")
    names.append("face_present")
    for hand in HAND_NAMES:
        for hand_ref in config.HAND_REFERENCE_POINTS:
            for face_ref in config.FACE_REFERENCE_POINTS:
                names.append(f"dist_{hand.lower()}_{hand_ref}_to_{face_ref}")
    for hand in HAND_NAMES:
        names.append(f"{hand.lower()}_hand_rel_x")
        names.append(f"{hand.lower()}_hand_rel_y")
    return names


FEATURE_NAMES = _build_feature_names()
FEATURE_LENGTH = len(FEATURE_NAMES)


def build_feature_vector(detection_result, image_w, image_h):
    """
    Converts one frame's detection result (from LandmarkDetector.process)
    into a flat np.float32 array of length FEATURE_LENGTH.

    Missing hands/face are represented with zeros plus a presence flag,
    exactly as required, so the vector length never changes.
    """
    vec = []

    raw_hands = {
        "Left": detection_result["hands"]["Left"],
        "Right": detection_result["hands"]["Right"],
    }

    # 1 + 2: normalized hand landmarks (63 each)
    for hand in HAND_NAMES:
        normalized = _normalize_hand(raw_hands[hand])
        if normalized is None:
            vec.extend([0.0] * (config.NUM_HAND_LANDMARKS * 3))
        else:
            vec.extend(normalized.flatten().tolist())

    # 3: presence flags
    for hand in HAND_NAMES:
        vec.append(1.0 if raw_hands[hand] is not None else 0.0)

    # 4: normalized face landmarks
    normalized_face = _normalize_face(detection_result["face"])
    if normalized_face is None:
        vec.extend([0.0] * (len(config.FACE_LANDMARK_NAMES) * 3))
    else:
        for name in config.FACE_LANDMARK_NAMES:
            vec.extend(normalized_face[name].tolist())

    # 5: face presence flag
    vec.append(1.0 if detection_result["face"] is not None else 0.0)

    # 6: hand-to-face distances (raw, un-normalized landmarks + pixel scale)
    for hand in HAND_NAMES:
        vec.extend(
            _hand_to_face_distances(
                raw_hands[hand], detection_result["face"], image_w, image_h
            )
        )

    # 7: relative hand position vs face
    for hand in HAND_NAMES:
        vec.extend(
            _relative_hand_position(
                raw_hands[hand], detection_result["face"], image_w, image_h
            )
        )

    arr = np.array(vec, dtype=np.float32)
    assert arr.shape[0] == FEATURE_LENGTH, (
        f"Feature vector length mismatch: got {arr.shape[0]}, "
        f"expected {FEATURE_LENGTH}"
    )
    return arr
