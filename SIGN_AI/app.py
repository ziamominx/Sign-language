import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from ui.widgets import STYLE
from ui.home import HomeScreen




from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class CompleteScreen(QWidget):
    train, home = Signal(), Signal()

    def __init__(self, n):
        super().__init__()
        l = QVBoxLayout(self); l.setAlignment(Qt.AlignCenter); l.setSpacing(16)
        t = QLabel("DATA COLLECTION COMPLETE"); t.setObjectName("title"); t.setAlignment(Qt.AlignCenter)
        s = QLabel(f"Your data has been saved. ({n} new sample{'s' if n != 1 else ''})")
        s.setObjectName("sub"); s.setAlignment(Qt.AlignCenter)
        a = QPushButton("TRAIN MODEL"); a.setObjectName("primary"); a.clicked.connect(self.train.emit)
        b = QPushButton("BACK TO HOME"); b.clicked.connect(self.home.emit)
        l.addWidget(t); l.addWidget(s); l.addSpacing(24)
        for x in (a, b):
            x.setFixedSize(420, 66); l.addWidget(x, alignment=Qt.AlignCenter)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignFlow — Live Sign Recognition")
        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)
        self.go_home()

    def _show(self, w):
        old = self.stack.currentWidget()
        if old is not None:
            if hasattr(old, "shutdown"):
                old.shutdown()
            self.stack.removeWidget(old); old.deleteLater()
        self.stack.addWidget(w); self.stack.setCurrentWidget(w); w.setFocus()

    def go_home(self):
        h = HomeScreen()
        h.add.connect(self.go_collect); h.test.connect(self.go_test); h.quit.connect(self.close)
        self._show(h)

    def go_collect(self):
        from ui.collect_screen import CollectScreen
        c = CollectScreen()
        c.back.connect(self.go_home); c.done.connect(self.go_complete)
        self._show(c)

    def go_complete(self, n):
        c = CompleteScreen(n)
        c.train.connect(self.go_train); c.home.connect(self.go_home)
        self._show(c)

    def go_train(self):
        from ui.train_screen import TrainScreen
        t = TrainScreen()
        t.finished_ok.connect(self.go_home); t.back.connect(self.go_home)
        self._show(t)

    def go_test(self):
        from ui.test_screen import TestScreen
        t = TestScreen(); t.back.connect(self.go_home)
        self._show(t)

    def keyPressEvent(self, e):
        w = self.stack.currentWidget()
        if hasattr(w, "key_pressed"):
            w.key_pressed(e)
        super().keyPressEvent(e)

    def closeEvent(self, e):
        w = self.stack.currentWidget()
        if hasattr(w, "shutdown"):
            w.shutdown()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow(); win.showMaximized()
    sys.exit(app.exec())
