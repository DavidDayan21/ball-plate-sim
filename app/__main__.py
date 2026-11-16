"""The software for the general public (phase D, in progress): python -m app

python -m app --self-check [FILE]: the check of the software as packaged (app/self_check.py), without its window.
"""
import sys

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication

from app import NAME
from app.i18n import Texts, language_of
from app.icon import WINDOWS_ID, icon
from app.self_check import OPTION, run
from app.settings import remembered, restore_geometry, settings
from app.window import MainWindow

SIZE = (1200, 750)          # (pixels) at the first launch, reduced to fit in the screen


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    if OPTION in argv:                      # the check of the package, then exit
        i = argv.index(OPTION)
        return run(argv[i + 1] if i + 1 < len(argv) else None)
    if sys.platform == "win32":             # the task bar of Windows then shows the icon of the software
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_ID)
    application = QApplication(argv)
    application.setApplicationName(NAME)
    application.setWindowIcon(icon())
    application.setStyle("Fusion")          # same base on Windows and Mac, under the style sheet of app/theme.py
    kept = settings()
    language, theme = remembered(kept)
    window = MainWindow(Texts(language or language_of(QLocale.system().name())), mode=theme, settings=kept)
    if not restore_geometry(window, kept):
        screen = application.primaryScreen().availableGeometry()
        window.resize(min(SIZE[0], int(0.9 * screen.width())), min(SIZE[1], int(0.9 * screen.height())))
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
