import collections
import json
import os
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
                               QLabel, QPushButton, QMessageBox)
from .widgets import VideoView, draw_hands, to_qimage

MAX_SENTENCE_WORDS = 12
NO_HANDS_CLEAR_DELAY = 0.5


def load_artifacts():
    from utils import config
    from test_model import SignModel
    from utils.inference import load_inference_pickle
    for p in (config.CONFIG_FILE, config.LABEL_ENCODER_FILE, config.SCALER_FILE):
        if not os.path.exists(p):
            raise RuntimeError("No trained model found. Add a sign and train first.")
    with open(config.CONFIG_FILE) as f:
        rc = json.load(f)
    # config.json may hold an absolute path from another machine -> use local path
    rc["model_path"] = config.MODEL_FILE_KERAS if rc["model_type"] == "lstm" else config.MODEL_FILE_SKLEARN
    le = load_inference_pickle(config.LABEL_ENCODER_FILE)
    scaler = load_inference_pickle(config.SCALER_FILE)
    return SignModel(rc), le, scaler, rc


class TestWorker(QThread):
    """Same recognition/subtitle loop as test_model.main(), minus cv2 windows."""
    frame = Signal(QImage)
    state = Signal(str, str)     # detected label, subtitle
    ready = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = self._clear = False

    def stop(self): self._stop = True
    def clear(self): self._clear = True

    def run(self):
        try:
            from utils import config
            from utils.features import build_feature_vector
            from utils.landmarks import LandmarkDetector
            from utils.preprocessing import resample_sequence
            from test_model import apply_fillers, split_class_name
            model, le, scaler, rc = load_artifacts()
        except Exception as ex:
            self.error.emit(str(ex)); return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Could not open webcam."); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        detector = LandmarkDetector()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        seq_len, sep = rc["sequence_len"], rc.get("class_separator", "::")
        fbuf = collections.deque(maxlen=config.MAX_SEQUENCE_LEN)
        pbuf = collections.deque(maxlen=config.PREDICTION_BUFFER_SIZE)
        interval = 1.0 / config.TARGET_CAPTURE_FPS
        last_cap, no_hands = 0.0, None
        label, words, last_word, last_emit = "", [], None, None
        self.ready.emit()
        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Camera read failed."); break
                now = time.monotonic()
                if now - last_cap < interval:
                    continue
                last_cap = now
                frame = cv2.flip(frame, 1)
                det = detector.process(frame)
                hands = det["hands"]["Left"] is not None or det["hands"]["Right"] is not None
                if self._clear:                      # subtitle X button: full reset
                    self._clear = False
                    words, last_word, label = [], None, ""
                    fbuf.clear(); pbuf.clear()
                if hands:
                    fbuf.append(build_feature_vector(det, w, h))
                if not hands:
                    if no_hands is None:
                        no_hands = now
                    elif now - no_hands > NO_HANDS_CLEAR_DELAY:
                        fbuf.clear(); pbuf.clear(); label, last_word = "", None
                else:
                    no_hands = None

                if len(fbuf) >= config.MIN_SEQUENCE_LEN:
                    fixed = resample_sequence(np.stack(list(fbuf), axis=0), target_len=seq_len)
                    probs = model.predict_proba(scaler.transform(fixed))
                    i = int(np.argmax(probs)); conf = float(probs[i])   # confidence still used
                    pbuf.append((le.classes_[i] if conf >= config.MIN_CONFIDENCE_THRESHOLD else None, conf))
                    votes = collections.Counter(c for c, _ in pbuf if c is not None)
                    label = ""
                    if votes:
                        best, count = votes.most_common(1)[0]
                        if count >= config.MIN_CONSECUTIVE_AGREEMENT:
                            _, lab = split_class_name(best, sep)
                            if not lab.startswith("NOT_"):
                                label = lab
                    if label and label != last_word:
                        words = [label] if len(words) >= MAX_SENTENCE_WORDS else words + [label]
                        last_word = label
                    elif not label:
                        last_word = None

                cur = (label, " ".join(apply_fillers(words)))
                if cur != last_emit:
                    self.state.emit(*cur); last_emit = cur
                draw_hands(frame, det)               # face landmarks intentionally not drawn
                self.frame.emit(to_qimage(frame))
        finally:
            cap.release()
            detector.close()


class TestScreen(QWidget):
    back = Signal()

    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0)
        self.video = VideoView(); l.addWidget(self.video)
        v = QVBoxLayout(self.video); v.setContentsMargins(16, 16, 16, 16)
        bk = QPushButton("←  BACK"); bk.setObjectName("small"); bk.setCursor(Qt.PointingHandCursor)
        bk.setFocusPolicy(Qt.NoFocus); bk.clicked.connect(self._back)
        row = QHBoxLayout(); row.addWidget(bk); row.addStretch(1); v.addLayout(row)
        self.det = QLabel("STARTING CAMERA..."); self.det.setObjectName("panel")
        v.addWidget(self.det, alignment=Qt.AlignLeft)
        v.addStretch(1)
        box = QFrame(); box.setObjectName("subtitleBox")
        box.setStyleSheet("QFrame#subtitleBox{background:#17273b;border:1px solid #34465c;border-radius:12px;}")
        box.setMinimumHeight(100)
        g = QGridLayout(box); g.setContentsMargins(20, 10, 20, 16)
        self.sub = QLabel("Show a sign to begin"); self.sub.setAlignment(Qt.AlignCenter); self.sub.setWordWrap(True)
        self.sub.setStyleSheet("font-size:32px;font-weight:700;color:#eef4fb;")
        x = QPushButton("✕"); x.setObjectName("x"); x.setFocusPolicy(Qt.NoFocus)
        x.setCursor(Qt.PointingHandCursor); x.clicked.connect(lambda: self.w and self.w.clear())
        g.addWidget(self.sub, 0, 0, 2, 1); g.addWidget(x, 0, 0, alignment=Qt.AlignTop | Qt.AlignRight)
        v.addWidget(box)
        self.w = TestWorker()
        self.w.frame.connect(self.video.set_image)
        self.w.state.connect(self._state)
        self.w.ready.connect(lambda: self._state("", ""))
        self.w.error.connect(self._error)
        self.w.start()

    def _state(self, label, subtitle):
        self.det.setText(f"SIGN DETECTED  ·  {label}" if label else "WAITING FOR SIGN")
        self.sub.setText(subtitle or "Show a sign to begin")

    def _error(self, msg):
        QMessageBox.warning(self, "Test", msg)
        self._back()

    def _back(self):
        self.shutdown(); self.back.emit()

    def shutdown(self):
        if self.w:
            self.w.stop(); self.w.wait(5000); self.w = None
