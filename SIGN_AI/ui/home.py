from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class HomeScreen(QWidget):
    add, test, quit = Signal(), Signal(), Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)
        eyebrow = QLabel("LIVE SIGN RECOGNITION"); eyebrow.setObjectName("eyebrow")
        eyebrow.setAlignment(Qt.AlignCenter)
        t = QLabel("Make every sign count."); t.setObjectName("title"); t.setAlignment(Qt.AlignCenter)
        s = QLabel("Recognize hand gestures and build a sentence in real time."); s.setObjectName("sub")
        s.setAlignment(Qt.AlignCenter)
        lay.addWidget(eyebrow); lay.addWidget(t); lay.addWidget(s); lay.addSpacing(30)
        for text, sig, name in (("START LIVE DEMO", self.test, "primary"),
                                ("ADD NEW SIGN", self.add, ""), ("QUIT", self.quit, "danger")):
            b = QPushButton(text); b.setObjectName(name); b.setFixedSize(420, 66)
            b.setCursor(Qt.PointingHandCursor); b.clicked.connect(sig.emit)
            lay.addWidget(b, alignment=Qt.AlignCenter)
        lay.addSpacing(18)
        hint = QLabel("Tip: face the camera and hold each sign clearly.")
        hint.setObjectName("hint"); hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)
