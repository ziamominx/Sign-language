"""
Thin wrapper around MediaPipe's Tasks API: HandLandmarker + FaceLandmarker.

Why this uses the Tasks API instead of the old `mp.solutions.hands` /
`mp.solutions.face_mesh` API: current `mediapipe` releases on PyPI
(0.10.30+) removed that legacy "solutions" API entirely. The old API only
exists in `mediapipe==0.10.14` and earlier, which only ships wheels for
Python <=3.12 -- if pip can't find that version for you, it's because your
Python is newer than 3.12, and this Tasks-API version is what you need.

Responsible only for:
  - downloading the two small model bundles the first time they're needed
  - running detection on a BGR frame
  - returning landmarks in a simple, consistent Python structure (same
    shape regardless of which MediaPipe API is used underneath)
  - drawing the skeleton / mesh for visualization

Feature engineering (normalization, distances, etc.) lives in features.py,
NOT here, so collection and testing always share the exact same pipeline.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from . import config

drawing_utils = mp_vision.drawing_utils
drawing_styles = mp_vision.drawing_styles
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
FACE_TESSELATION = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
FACE_CONTOURS = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS


def _download_if_missing(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    print(f"Downloading MediaPipe model to {dest_path} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download required MediaPipe model from {url}\n"
            f"({exc})\n"
            "If your network blocks storage.googleapis.com, download the "
            f"file manually from that URL and save it to:\n  {dest_path}"
        ) from exc


class LandmarkDetector:
    """Owns the MediaPipe HandLandmarker + FaceLandmarker task instances."""

    def __init__(self):
        _download_if_missing(config.HAND_LANDMARKER_MODEL_URL,
                              config.HAND_LANDMARKER_MODEL_PATH)
        _download_if_missing(config.FACE_LANDMARKER_MODEL_URL,
                              config.FACE_LANDMARKER_MODEL_PATH)

        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.FACE_LANDMARKER_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

        self._start_time = time.time()

    def _timestamp_ms(self):
        # VIDEO running mode requires a monotonically increasing timestamp.
        return int((time.time() - self._start_time) * 1000)

    def process(self, frame_bgr):
        """
        Runs hand + face detection on one BGR frame.

        Returns a dict:
            {
              "hands": {"Left": [(x,y,z), ...21] or None,
                        "Right": [(x,y,z), ...21] or None},
              "hands_raw": {"landmarks": [[NormalizedLandmark, ...21], ...],
                            "labels": ["Left"/"Right", ...]}  (for drawing),
              "face": [(x,y,z), ...478] or None,
              "face_raw": [NormalizedLandmark, ...478] or None  (for drawing),
            }
        Coordinates are MediaPipe's native normalized (0..1) image
        coordinates, x/y relative to image width/height.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()

        hand_result = self.hand_landmarker.detect_for_video(mp_image, ts)
        face_result = self.face_landmarker.detect_for_video(mp_image, ts)

        hands_out = {"Left": None, "Right": None}
        hands_raw_landmarks = []
        hands_raw_labels = []
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness[0].category_name  # "Left" / "Right"
                pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                hands_out[label] = pts
                hands_raw_landmarks.append(hand_landmarks)
                hands_raw_labels.append(label)

        face_out = None
        face_raw_landmarks = None
        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            face_out = [(lm.x, lm.y, lm.z) for lm in face_landmarks]
            face_raw_landmarks = face_landmarks

        return {
            "hands": hands_out,
            "hands_raw": {"landmarks": hands_raw_landmarks, "labels": hands_raw_labels},
            "face": face_out,
            "face_raw": face_raw_landmarks,
        }

    def close(self):
        self.hand_landmarker.close()
        self.face_landmarker.close()


def draw_landmarks(frame_bgr, detection_result):
    """Draws hand skeletons and the face mesh contour onto the frame in-place."""
    for hand_landmarks in detection_result["hands_raw"]["landmarks"]:
        drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

    face_landmarks = detection_result["face_raw"]
    if face_landmarks is not None:
        drawing_utils.draw_landmarks(
            frame_bgr,
            face_landmarks,
            FACE_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            frame_bgr,
            face_landmarks,
            FACE_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )


def hand_bounding_box(hand_points, image_w, image_h, margin=20):
    """Pixel-space bounding box (x1, y1, x2, y2) around a hand's landmarks."""
    xs = [p[0] * image_w for p in hand_points]
    ys = [p[1] * image_h for p in hand_points]
    x1, x2 = max(int(min(xs)) - margin, 0), min(int(max(xs)) + margin, image_w)
    y1, y2 = max(int(min(ys)) - margin, 0), min(int(max(ys)) + margin, image_h)
    return x1, y1, x2, y2
