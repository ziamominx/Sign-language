import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

STYLE = """
QWidget{background:#0b1220;color:#eef4fb;font-family:'Segoe UI';font-size:16px;}
QLabel{background:transparent;}
QPushButton{background:#1b2a40;border:1px solid #34465c;border-radius:12px;padding:14px 28px;font-size:19px;font-weight:600;}
QPushButton:hover{background:#253c55;border-color:#6b91aa;}
QPushButton:pressed{background:#34516a;}
QPushButton:focus{border:2px solid #6fe2ce;}
QPushButton#primary{background:#54d6bd;color:#092329;border:1px solid #54d6bd;}
QPushButton#primary:hover{background:#80e9d4;}
QPushButton#danger:hover{background:#c0392b;}
QPushButton#small{padding:9px 17px;font-size:15px;border-radius:9px;background:#17273b;}
QPushButton#small:hover{background:#29445d;}
QPushButton#x{padding:4px 11px;font-size:16px;border-radius:8px;background:#26384d;}
QPushButton#x:hover{background:#c0392b;}
QLineEdit{background:#17273b;border:2px solid #34465c;border-radius:10px;padding:12px 16px;font-size:20px;}
QLineEdit:focus{border-color:#54d6bd;}
QLabel#title{font-size:48px;font-weight:700;letter-spacing:1px;}
QLabel#h2{font-size:30px;font-weight:700;}
QLabel#sub{color:#a9bbcd;font-size:18px;}
QLabel#eyebrow{color:#54d6bd;font-size:15px;font-weight:700;letter-spacing:2px;}
QLabel#hint{color:#a9bbcd;font-size:15px;}
QProgressBar{background:#1b2a40;border:none;border-radius:6px;max-height:12px;min-height:12px;}
QProgressBar::chunk{background:#54d6bd;border-radius:6px;}
#video{background:#03070c;}
#panel{background:#17273b;border:1px solid #34465c;border-radius:10px;padding:12px 18px;font-size:17px;}
"""


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def draw_hands(frame, det):
    """Hand skeletons only -- face landmarks are still detected/used, just not drawn."""
    from utils.landmarks import drawing_utils, drawing_styles, HAND_CONNECTIONS
    landmark_style = drawing_styles.get_default_hand_landmarks_style()
    connection_style = drawing_styles.get_default_hand_connections_style()
    for lm in det["hands_raw"]["landmarks"]:
        drawing_utils.draw_landmarks(
            frame, lm, HAND_CONNECTIONS,
            landmark_style, connection_style)


class VideoView(QLabel):
    """Aspect-preserving video area; child widgets/layouts act as overlays."""
    def __init__(self):
        super().__init__()
        self.setObjectName("video")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._img = None

    def set_image(self, img):
        self._img = img
        self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _render(self):
        if self._img is not None:
            self.setPixmap(QPixmap.fromImage(self._img).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
