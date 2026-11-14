"""The software for the general public (phase D, in progress): python -m app"""
import sys

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication

from app.i18n import Texts, language_of
from app.window import MainWindow

SIZE = (1200, 750)          # (pixels) at startup, reduced to fit in the screen


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("ball-plate-sim")
    application.setStyle("Fusion")          # same base on Windows and Mac, under the style sheet of app/theme.py
    window = MainWindow(Texts(language_of(QLocale.system().name())))
    screen = application.primaryScreen().availableGeometry()
    window.resize(min(SIZE[0], int(0.9 * screen.width())), min(SIZE[1], int(0.9 * screen.height())))
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
