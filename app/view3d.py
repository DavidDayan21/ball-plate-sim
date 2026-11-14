"""The robot in 3D: the display scene (app/scene.py) rendered off screen by MuJoCo, drawn in a Qt widget with rounded
corners, with a few texts over it (Overlay) and the x and y axes of the plate.

Mouse: left drag turns the view around the plate, the wheel zooms.
The image has the size of the widget in physical pixels (high-resolution screens), up to scene.MAX_SIZE. After a
change of size, the renderer is created again at the new size once the size has stopped changing for RESIZE_DELAY;
until then, the last image is stretched.
"""
from dataclasses import dataclass, field
import math

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from app.scene import MAX_SIZE, DisplayScene

RESIZE_DELAY = 150              # (ms)
DEGREES_PER_PIXEL = 0.3         # turn of the view per pixel of mouse drag
ZOOM_PER_NOTCH = 0.9            # distance multiplied by this per notch of the wheel (forward: closer)
DISTANCE = (0.25, 2.0)          # (m) closest and farthest view
ELEVATION = (-89.0, -5.0)       # (°) from straight above to almost level with the plate
RADIUS = 8                      # (pixels) corners of the view
AXIS_LENGTH = 28                # (pixels) axes of the plate, drawn in the corner


class Camera:
    """Point of view around the plate: azimuth and elevation (°), distance (m), point looked at. Without Qt."""

    def __init__(self, lookat_height: float):
        self.azimuth, self.elevation, self.distance = 90.0, -27.0, 0.68
        self.lookat = (0.0, 0.0, lookat_height)

    def turn(self, dx: float, dy: float) -> None:
        """Mouse drag of (dx, dy) pixels: to the right, the scene turns to the right; downward, the scene is
        seen from higher up."""
        self.azimuth = (self.azimuth - DEGREES_PER_PIXEL * dx) % 360.0
        self.elevation = min(max(self.elevation - DEGREES_PER_PIXEL * dy, ELEVATION[0]), ELEVATION[1])

    def zoom(self, notches: float) -> None:
        """notches > 0: closer."""
        self.distance = min(max(self.distance * ZOOM_PER_NOTCH ** notches, DISTANCE[0]), DISTANCE[1])

    def mujoco(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = self.azimuth, self.elevation, self.distance
        cam.lookat[:] = self.lookat
        return cam

    def screen_axes(self) -> tuple:
        """Directions on the screen (right, up) of the x and y axes of the world: ((xr, xu), (yr, yu)).
        The camera looks along f = (cos e·cos a, cos e·sin a, sin e); right r = (sin a, -cos a, 0), up = r × f."""
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        return ((math.sin(a), -math.cos(a) * math.sin(e)), (-math.cos(a), -math.sin(a) * math.sin(e)))


def image_size(width: int, height: int) -> tuple:
    """Size of the image rendered for a widget of width × height physical pixels: the same, reduced if needed to fit
    in MAX_SIZE (same proportions), at least 1 × 1."""
    f = min(1.0, MAX_SIZE[0] / max(width, 1), MAX_SIZE[1] / max(height, 1))
    return max(1, int(width * f)), max(1, int(height * f))


@dataclass
class Overlay:
    """What is written over the 3D view, and the colors of the theme (keys of app/theme.py)."""
    state: str = ""
    state_color: str = "#16A34A"
    hint: str = ""
    bottom_left: str = ""
    bottom_right: str = ""
    colors: dict = field(default_factory=dict)


class View3D(QWidget):
    def __init__(self, scene: DisplayScene, physics_data, lookat_height: float, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.physics_data = physics_data
        self.camera = Camera(lookat_height)
        self.overlay = Overlay()
        self.renderer = None
        self.pixels = None             # last image (height, width, 3), kept alive for self.image
        self.image = None
        self._drag = None              # last mouse position of a drag
        self._resize_timer = QTimer(self, singleShot=True, interval=RESIZE_DELAY)
        self._resize_timer.timeout.connect(self._create_renderer)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_scene(self, scene: DisplayScene) -> None:
        """Another display scene (theme changed): the renderer is created again for it."""
        self.close_renderer()
        self.scene = scene

    # --- Rendering ---------------------------------------------------------------------------
    def _physical_size(self) -> tuple:
        ratio = self.devicePixelRatioF()
        return image_size(round(self.width() * ratio), round(self.height() * ratio))

    def _create_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        width, height = self._physical_size()
        self.renderer = mujoco.Renderer(self.scene.model, height, width)

    def render_frame(self) -> None:
        """Renders the robot as it is now in the physics, then asks Qt to draw it."""
        if self.renderer is None:
            self._create_renderer()
        self.scene.follow(self.physics_data)
        self.renderer.update_scene(self.scene.data, camera=self.camera.mujoco())
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1     # the floor fades into the background
        self.pixels = np.ascontiguousarray(self.renderer.render())
        height, width, _ = self.pixels.shape
        self.image = QImage(self.pixels.data, width, height, 3 * width, QImage.Format_RGB888)
        self.update()

    def close_renderer(self) -> None:
        self._resize_timer.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def resizeEvent(self, event) -> None:
        if self.renderer is not None:
            self._resize_timer.start()        # restarted at every change of size
        super().resizeEvent(event)

    # --- Drawing -----------------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        o, c = self.overlay, self.overlay.colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, RADIUS, RADIUS)
        p.setClipPath(frame)
        if self.image is None:
            p.fillRect(rect, QColor(c.get("bg", "#000000")))
        else:
            p.drawImage(rect, self.image)
        p.setClipping(False)
        if c:
            p.setPen(QPen(QColor(c["border"]), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
            self._chip(p, o.state, o.state_color, c)
            small = QFont(self.font())
            small.setPixelSize(12)
            p.setFont(small)
            p.setPen(QColor(c["text2"]))
            text = rect.adjusted(16, 0, -16, -12)
            p.drawText(rect.adjusted(16, 22, -16, 0), Qt.AlignTop | Qt.AlignRight, o.hint)
            p.drawText(text, Qt.AlignBottom | Qt.AlignLeft, o.bottom_left)
            p.drawText(text, Qt.AlignBottom | Qt.AlignRight, o.bottom_right)
            self._axes(p, QPointF(rect.width() - 44, rect.height() - 64), c)
        p.end()

    def _chip(self, p: QPainter, text: str, dot: str, c: dict) -> None:
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        r = QRectF(16, 16, p.fontMetrics().horizontalAdvance(text) + 36, 28)
        background = QColor(c["surface"])
        background.setAlpha(230)
        p.setPen(QPen(QColor(c["border"]), 1))
        p.setBrush(background)
        p.drawRoundedRect(r, 14, 14)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(dot))
        p.drawEllipse(QPointF(r.x() + 15, r.center().y()), 4, 4)
        p.setPen(QColor(c["text"]))
        p.drawText(r.adjusted(26, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def _axes(self, p: QPainter, origin: QPointF, c: dict) -> None:
        """The x and y axes of the plate as seen from the camera (they turn with the view)."""
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        for (right, up), name, color in zip(self.camera.screen_axes(), "xy", (c["axis_x"], c["axis_y"])):
            end = origin + QPointF(AXIS_LENGTH * right, -AXIS_LENGTH * up)
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(origin, end)
            p.drawText(origin + QPointF((AXIS_LENGTH + 9) * right - 4, -(AXIS_LENGTH + 9) * up + 4), name)

    # --- Mouse -------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            q = event.position()
            self.camera.turn(q.x() - self._drag.x(), q.y() - self._drag.y())
            self._drag = q

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = None

    def wheelEvent(self, event) -> None:
        self.camera.zoom(event.angleDelta().y() / 120)       # 120 per notch (Qt)
