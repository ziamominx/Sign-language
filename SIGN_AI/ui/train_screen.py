import re
import sys
from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton
from utils import config

STAGES = ["Loading dataset", "Preparing sequences", "Training model",
          "Validating model", "Saving model"]
MARKERS = [("Saving model", 4), ("Validation accuracy", 3), ("Test set report", 3),
           ("Training model", 2), ("Training sequences", 1), ("Classes found", 1)]


class TrainScreen(QWidget):
    finished_ok = Signal()
    back = Signal()

    def __init__(self):
        super().__init__()
        l = QVBoxLayout(self); l.setAlignment(Qt.AlignCenter); l.setSpacing(14)
        self.title = QLabel("TRAINING MODEL"); self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignCenter)
        self.bar = QProgressBar(); self.bar.setRange(0, 0); self.bar.setTextVisible(False)
        self.bar.setFixedWidth(520)
        self.stage_lbl = [QLabel() for _ in STAGES]
        self.note = QLabel("Please wait..."); self.note.setObjectName("sub")
        self.note.setAlignment(Qt.AlignCenter); self.note.setWordWrap(True)
        self.btn = QPushButton("BACK TO HOME"); self.btn.setFixedSize(300, 56); self.btn.hide()
        self.btn.clicked.connect(self.back.emit)
        l.addWidget(self.title); l.addSpacing(20)
        l.addWidget(self.bar, alignment=Qt.AlignCenter); l.addSpacing(10)
        for s in self.stage_lbl:
            s.setAlignment(Qt.AlignCenter); l.addWidget(s)
        l.addSpacing(10); l.addWidget(self.note); l.addWidget(self.btn, alignment=Qt.AlignCenter)
        self.stage, self.log = 0, ""
        self._paint()
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(config.PROJECT_ROOT)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(sys.executable, ["-u", "train_model.py"])   # existing script, unchanged

    def _paint(self, complete=False):
        for i, (lab, name) in enumerate(zip(self.stage_lbl, STAGES)):
            if complete or i < self.stage:
                lab.setText(f"✓  {name}"); lab.setStyleSheet("color:#2ecc71;font-size:18px;")
            elif i == self.stage:
                lab.setText(f"●  {name}..."); lab.setStyleSheet("color:#fff;font-size:20px;font-weight:700;")
            else:
                lab.setText(f"○  {name}"); lab.setStyleSheet("color:#5b6580;font-size:18px;")

    def _read(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        self.log += text
        for line in text.splitlines():
            for key, idx in MARKERS:
                if key in line and idx > self.stage:
                    self.stage = idx; self._paint()
            m = re.search(r"Epoch (\d+)/(\d+)", line)
            if m:
                self.note.setText(f"Epoch {m.group(1)} (up to {m.group(2)}, may stop early)")

    def _done(self, code, _status):
        if code == 0:
            self._paint(complete=True)
            self.bar.setRange(0, 1); self.bar.setValue(1)
            self.title.setText("MODEL TRAINING COMPLETE ✓")
            self.note.setText("Returning home...")
            QTimer.singleShot(1500, self.finished_ok.emit)
        else:
            self.bar.setRange(0, 1); self.bar.setValue(0)
            self.title.setText("TRAINING FAILED")
            self.note.setText("\n".join(self.log.strip().splitlines()[-8:]))
            self.btn.show()

    def shutdown(self):
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill(); self.proc.waitForFinished(2000)
