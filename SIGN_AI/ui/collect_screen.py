import time
import uuid
import cv2
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QLineEdit, QStackedWidget, QMessageBox)
from utils import config
from .widgets import VideoView, draw_hands, to_qimage


class CollectWorker(QThread):
    """Same recording rules as collect_data.record_sequence, but non-blocking."""
    frame = Signal(QImage)
    progress = Signal(float, int)      # elapsed, frames (elapsed < 0 => idle)
    saved = Signal(str, int, int)      # label, frames, total samples
    status = Signal(str)
    error = Signal(str)

    def __init__(self, language, label):
        super().__init__()
        self.language, self.label = language, label
        self._cmd = None
        self._stop = self._stop_rec = False
        self.recording = False
        self.count = 0

    def request(self, kind):
        if not self.recording and self._cmd is None:
            self._cmd = kind
            return True
        return False

    def stop_early(self): self._stop_rec = True
    def stop(self): self._stop = True

    def run(self):
        try:
            from utils.features import build_feature_vector
            from utils.landmarks import LandmarkDetector
            from collect_data import ensure_csv_headers, append_sequence_to_csv
            ensure_csv_headers()
        except Exception as ex:
            self.error.emit(str(ex))
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        detector = LandmarkDetector()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        interval = 1.0 / config.TARGET_CAPTURE_FPS
        last_processed = 0.0
        rec = None
        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Camera read failed.")
                    break
                now = time.monotonic()
                if now - last_processed < interval:
                    continue
                last_processed = now
                frame = cv2.flip(frame, 1)
                det = detector.process(frame)
                if rec is None and self._cmd:
                    kind, self._cmd = self._cmd, None
                    rec = dict(label=self.label if kind == 1 else f"NOT_{self.label}",
                               start=now, last=0.0, vecs=[])
                    self.recording, self._stop_rec = True, False
                if rec:
                    el = now - rec["start"]
                    if el - rec["last"] >= interval:
                        rec["vecs"].append(build_feature_vector(det, w, h))
                        rec["last"] = el
                    self.progress.emit(el, len(rec["vecs"]))
                    if (self._stop_rec or el >= config.RECORDING_DURATION_SECONDS
                            or len(rec["vecs"]) >= config.MAX_SEQUENCE_LEN):
                        vecs = rec["vecs"]
                        if len(vecs) < config.MIN_SEQUENCE_LEN:
                            self.status.emit("Sequence rejected: too few frames")
                        else:
                            append_sequence_to_csv(self.language, rec["label"],
                                                   uuid.uuid4().hex[:12], vecs)
                            self.count += 1
                            self.saved.emit(rec["label"], len(vecs), self.count)
                        rec, self.recording = None, False
                        self.progress.emit(-1, 0)
                draw_hands(frame, det)
                self.frame.emit(to_qimage(frame))
        finally:
            cap.release()
            detector.close()


class CollectScreen(QWidget):
    done = Signal(int)
    back = Signal()

    def __init__(self):
        super().__init__()
        self.w = None
        self.stack = QStackedWidget()
        QVBoxLayout(self).addWidget(self.stack)
        self.stack.addWidget(self._setup_page())
        self.stack.addWidget(self._camera_page())

    def _setup_page(self):
        p = QWidget(); l = QVBoxLayout(p); l.setAlignment(Qt.AlignCenter); l.setSpacing(16)
        t = QLabel("Which sign language are you recording?"); t.setObjectName("h2")
        t.setAlignment(Qt.AlignCenter)
        self.lang = QLineEdit(); self.lang.setPlaceholderText("Enter language (e.g. ASL)")
        self.sign = QLineEdit(); self.sign.setPlaceholderText("Sign to record (e.g. FATHER)")
        for e in (self.lang, self.sign):
            e.setFixedWidth(460); e.returnPressed.connect(self._start)
        go = QPushButton("START CAMERA"); go.setObjectName("primary"); go.setFixedSize(460, 60)
        go.clicked.connect(self._start)
        bk = QPushButton("BACK"); bk.setFixedSize(460, 56); bk.clicked.connect(self.back.emit)
        for x in (t, self.lang, self.sign, go, bk):
            l.addWidget(x, alignment=Qt.AlignCenter)
        return p

    def _camera_page(self):
        p = QWidget(); l = QVBoxLayout(p); l.setContentsMargins(0, 0, 0, 0)
        self.video = VideoView(); l.addWidget(self.video)
        v = QVBoxLayout(self.video); v.setContentsMargins(16, 16, 16, 16)
        top = QHBoxLayout(); v.addLayout(top); v.addStretch(1)
        self.panel = QLabel(); self.panel.setObjectName("panel")
        self.recl = QLabel(); self.recl.setObjectName("panel"); self.recl.hide()
        top.addWidget(self.panel, alignment=Qt.AlignTop); top.addStretch(1)
        top.addWidget(self.recl, alignment=Qt.AlignTop)
        self.toast = QLabel(); self.toast.setObjectName("panel"); self.toast.hide()
        v.addWidget(self.toast, alignment=Qt.AlignHCenter)
        self._toast_timer = QTimer(self, singleShot=True, timeout=self.toast.hide)
        return p

    def _panel_text(self, n):
        k = "<b style='color:#4f7cff'>%s</b>"
        self.panel.setText(
            f"<div style='font-size:20px'><b>Recording Sign: {self.sign_name}</b></div>"
            f"{k % '[1]'} Positive Sample<br>{k % '[2]'} Negative Sample<br>"
            f"{k % '[Q]'} Finish<br><span style='color:#8b95b0'>Samples: {n}</span>")

    def _start(self):
        lang, sign = self.lang.text().strip(), self.sign.text().strip().upper()
        if not lang or not sign:
            QMessageBox.information(self, "Missing info", "Enter both the language and the sign.")
            return
        self.sign_name = sign
        self._panel_text(0)
        self.w = CollectWorker(lang, sign)
        self.w.frame.connect(self.video.set_image)
        self.w.progress.connect(self._progress)
        self.w.saved.connect(lambda lab, n, tot: (self._panel_text(tot),
                             self._show_toast(f"Saved {lab} ({n} frames)", "#8b95b0")))
        self.w.status.connect(lambda m: self._show_toast(m, "#e74c3c"))
        self.w.error.connect(self._error)
        self.w.start()
        self.stack.setCurrentIndex(1)

    def _progress(self, el, n):
        if el < 0:
            self.recl.hide()
        else:
            self.recl.setText(f"<span style='color:#ff4d4d'>●</span> RECORDING "
                              f"{el:.1f} / {config.RECORDING_DURATION_SECONDS:.1f}s · {n} frames"
                              f"<br><span style='color:#8b95b0'>[S] stop early</span>")
            self.recl.show()

    def _show_toast(self, text, color):
        self.toast.setText(f"<span style='font-size:24px;color:{color}'><b>{text}</b></span>")
        self.toast.show(); self._toast_timer.start(1400)

    def _error(self, msg):
        QMessageBox.warning(self, "Camera", msg)
        self.shutdown(); self.back.emit()

    def key_pressed(self, e):
        if self.w is None:
            return
        k = e.key()
        if k in (Qt.Key_1, Qt.Key_2):
            if self.w.request(1 if k == Qt.Key_1 else 2):
                self._show_toast("✓ POSITIVE SAMPLE" if k == Qt.Key_1 else "✓ NEGATIVE SAMPLE",
                                 "#2ecc71" if k == Qt.Key_1 else "#f5a623")
        elif k == Qt.Key_S:
            self.w.stop_early()
        elif k == Qt.Key_Q:
            n = self.w.count
            self.shutdown(); self.done.emit(n)

    def shutdown(self):
        if self.w:
            self.w.stop(); self.w.wait(4000); self.w = None
