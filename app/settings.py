"""What the software remembers from one launch to the next, with Qt (QSettings: the registry on Windows, a file of
preferences on Mac): the theme and the language once the user has chosen them (until then, those of the system), the
size and position of the window. Not the brain, the ball, the friction or the latency: every launch starts from the
nominal conditions, so that comparisons stay readable.
"""
from PySide6.QtCore import QByteArray, QSettings

from app.i18n import LANGUAGES
from app.theme import MODES

ORGANIZATION, APPLICATION = "ball-plate-sim", "app"   # where they are kept: the same if the software is renamed


def settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def remembered(s: QSettings) -> tuple:
    """(language, theme) chosen at a previous launch; None for one never chosen, or unknown to this version."""
    language, theme = s.value("language"), s.value("theme")
    return language if language in LANGUAGES else None, theme if theme in MODES else None


def restore_geometry(window, s: QSettings) -> bool:
    """The size and position of the window at its last closing; False if there are none."""
    geometry = s.value("geometry")
    return isinstance(geometry, QByteArray) and window.restoreGeometry(geometry)
