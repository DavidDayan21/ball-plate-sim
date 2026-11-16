"""The icon of the software, drawn by code at every size it is used (sharp from 16 to 512 pixels): the orange ball
on the tilted plate, carried by two rods from the base, on a dark rounded square (dark: it stands out on light and
dark task bars). The geometry is that of a square of side 1."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QRadialGradient

SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
COLORS = dict(background="#1E293B", plate="#CBD5E1", base="#64748B", ball="#EA580C")
TILT = math.radians(8)                  # the plate, higher on the right
CENTER = (0.5, 0.47)                    # middle of the plate
HALF, THICKNESS = 0.36, 0.064           # half length and thickness of the plate
BALL = 0.125                            # radius of the ball
WINDOWS_ID = "ball-plate-sim.app"       # for Windows: the task bar shows this icon, not that of Python


def _on_plate(t: float, n: float = 0.0) -> QPointF:
    """Point at t along the plate from its middle, n across it (n < 0: above it)."""
    ux, uy = math.cos(TILT), -math.sin(TILT)
    return QPointF(CENTER[0] + t * ux - n * uy, CENTER[1] + t * uy + n * ux)


def draw(size: int) -> QImage:
    """The icon, size × size pixels, transparent outside its rounded square."""
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size, size)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(COLORS["background"]))
    p.drawRoundedRect(QRectF(0, 0, 1, 1), 0.22, 0.22)
    base = QColor(COLORS["base"])
    p.setBrush(base)
    p.drawRoundedRect(QRectF(0.20, 0.80, 0.60, 0.065), 0.032, 0.032)
    p.setPen(QPen(base, 0.042, Qt.SolidLine, Qt.RoundCap))
    for t, x in ((-0.22, 0.33), (0.22, 0.67)):                         # the rods, from the base to the plate
        p.drawLine(QPointF(x, 0.81), _on_plate(t, THICKNESS / 2))
    p.setPen(QPen(QColor(COLORS["plate"]), THICKNESS, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(_on_plate(-HALF), _on_plate(HALF))
    p.setPen(Qt.NoPen)
    center = _on_plate(-0.06, -(THICKNESS / 2 + BALL))                  # the ball, resting near the middle
    p.setBrush(QColor(COLORS["ball"]))
    p.drawEllipse(center, BALL, BALL)
    shine = QRadialGradient(center + QPointF(-0.045, -0.055), 0.07)      # a light from the upper left
    shine.setColorAt(0, QColor(255, 200, 160, 170))
    shine.setColorAt(1, QColor(255, 200, 160, 0))
    p.setBrush(shine)
    p.drawEllipse(center, BALL, BALL)
    p.end()
    return image


def icon() -> QIcon:
    result = QIcon()
    for size in SIZES:
        result.addPixmap(QPixmap.fromImage(draw(size)))
    return result


def ball_center() -> tuple:
    """Center of the ball in the square of side 1 (for the tests)."""
    c = _on_plate(-0.06, -(THICKNESS / 2 + BALL))
    return c.x(), c.y()
