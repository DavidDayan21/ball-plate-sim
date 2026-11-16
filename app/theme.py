"""Look of the software, in one place: the colors of the light and dark themes, the font, the style sheet of the
widgets and the colors of the 3D scene. The widgets only name their role (objectName), never a color.

Panel on the side and detailed measurements of an instrument; the orange of the ping-pong ball as the accent
color, kept for what is selected (brain, tool, setpoint, theme, language, pause): the buttons that act are
neutral. Font: Inter (app/fonts/, SIL Open Font License), shipped with the software so that it looks the same
on Windows and Mac.
"""
from pathlib import Path

from app.scene import SceneStyle

FONTS = Path(__file__).resolve().parent / "fonts"
FONT = "Inter"
RADIUS = 8              # (pixels) corners of the cards and of the 3D view
MODES = ("light", "dark")

COLORS = {
    "light": dict(bg="#EEF1F3", surface="#FFFFFF", surface2="#F1F3F5", border="#DDE1E6", text="#14161A",
                  text2="#5F6672", text3="#959CA7", accent="#EA580C", accent_text="#FFFFFF",
                  good="#16A34A", bad="#DC2626", axis_x="#E5484D", axis_y="#2F9E63",
                  brain_pid="#2563EB", brain_imitation="#7C3AED"),
    "dark": dict(bg="#0A0D10", surface="#11161B", surface2="#182027", border="#25303A", text="#ECEEF2",
                 text2="#A3AAB5", text3="#6B7380", accent="#FB7A3C", accent_text="#1C0D04",
                 good="#22C55E", bad="#F87171", axis_x="#F2555A", axis_y="#3DD68C",
                 brain_pid="#60A5FA", brain_imitation="#A78BFA"),
}


def rgb(color: str) -> tuple:
    """"#RRGGBB" -> (r, g, b) in 0..1."""
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


LIT_FLOOR = 1.15        # light theme: the lit floor comes out 1.15 times brighter than its color (measured in the
                        # default view): its color is darkened as much, so that it meets the background seamlessly


def scene_style(mode: str) -> SceneStyle:
    """Colors of the 3D scene: the floor continues the background of the window."""
    c = COLORS[mode]
    bg = rgb(c["bg"])
    if mode == "light":
        floor = tuple(v / LIT_FLOOR for v in bg)
        return SceneStyle(sky=bg, floor=floor, pedestal=rgb(c["surface"]),
                          motor=(0.14, 0.15, 0.17), metal=(0.85, 0.86, 0.88), arm=(0.32, 0.35, 0.39),
                          plate=(0.40, 0.45, 0.52), plate_alpha=0.38, ambient=0.45)   # smoked acrylic
    return SceneStyle(sky=bg, floor=bg, pedestal=rgb(c["surface2"]),
                      motor=(0.24, 0.26, 0.30), metal=(0.72, 0.74, 0.78), arm=(0.50, 0.53, 0.58),
                      plate=(0.80, 0.90, 1.0), plate_alpha=0.32, ambient=0.35)


def stylesheet(mode: str) -> str:
    """Style sheet (Qt) of the window, for the theme `mode`."""
    c = COLORS[mode]
    r, rs = RADIUS, RADIUS - 2
    return f"""
    QWidget {{ font-family: "{FONT}"; font-size: 13px; color: {c['text']}; }}
    QMainWindow, #window {{ background: {c['bg']}; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['text3']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}
    #card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {r}px; }}
    #title {{ font-size: 15px; font-weight: 600; }}
    #subtitle {{ color: {c['text2']}; font-size: 12px; }}
    #section {{ color: {c['text3']}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
    #caption {{ color: {c['text2']}; font-size: 12px; }}
    #readoutName {{ color: {c['text3']}; font-size: 11px; font-weight: 600; }}
    #readoutValue {{ font-size: 20px; font-weight: 600; }}
    #readoutUnit {{ color: {c['text2']}; font-size: 12px; }}
    #tableHead {{ font-size: 12px; font-weight: 600; }}
    #tableName {{ color: {c['text2']}; font-size: 12px; }}
    #tableValue {{ font-size: 13px; font-weight: 600; }}
    QPushButton {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {rs}px;
                   min-height: 34px; padding: 0 12px; font-weight: 500; }}
    QPushButton:hover {{ border-color: {c['text3']}; }}
    QPushButton#ghost {{ background: transparent; color: {c['text2']}; }}
    QPushButton#ghost:checked {{ color: {c['accent']}; border-color: {c['accent']}; }}
    QPushButton#tool:checked {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']};
                                font-weight: 600; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {c['surface']}; border: 2px solid {c['accent']}; width: 12px;
                                  margin: -6px 0; border-radius: 8px; }}
    #segment {{ background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: {rs}px; }}
    #segment QPushButton {{ background: transparent; border: none; color: {c['text2']}; min-height: 28px;
                            border-radius: {rs - 2}px; }}
    #segment QPushButton:checked {{ background: {c['accent']}; color: {c['accent_text']}; font-weight: 600; }}
    """


def load_fonts() -> None:
    """Makes the fonts of app/fonts/ available to Qt (once is enough; again does no harm)."""
    from PySide6.QtGui import QFontDatabase
    for path in sorted(FONTS.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(path))


def system_mode() -> str:
    """The theme of the system ("light" when it does not say)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    return "dark" if QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"
