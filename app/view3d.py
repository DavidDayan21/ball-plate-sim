"""The robot in 3D: MuJoCo renders the scene off screen, the image is drawn in a Qt widget.

Mouse: left drag turns the view around the plate, the wheel zooms.
The image has the size of the widget in physical pixels (high-resolution screens), up to MAX_SIZE. After a change
of size, the renderer is created again at the new size once the size has stopped changing for RESIZE_DELAY; until
then, the last image is stretched.
"""
import mujoco
import numpy as np
from PySide6.QtCore import QPointF, QRect, QTimer, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

MAX_SIZE = (2560, 1600)         # (width, height) of the largest image rendered, in pixels
RESIZE_DELAY = 150              # (ms)
DEGREES_PER_PIXEL = 0.3         # turn of the view per pixel of mouse drag
ZOOM_PER_NOTCH = 0.9            # distance multiplied by this per notch of the wheel (forward: closer)
DISTANCE = (0.25, 2.0)          # (m) closest and farthest view
ELEVATION = (-89.0, -5.0)       # (°) from straight above to almost level with the plate


class Camera:
    """Point of view around the plate: azimuth and elevation (°), distance (m), point looked at. Without Qt."""

    def __init__(self, lookat_height: float):
        self.azimuth, self.elevation, self.distance = 90.0, -35.0, 0.6     # those of the sandbox
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


def image_size(width: int, height: int) -> tuple:
    """Size of the image rendered for a widget of width × height physical pixels: the same, reduced if needed to fit
    in MAX_SIZE (same proportions), at least 1 × 1."""
    f = min(1.0, MAX_SIZE[0] / max(width, 1), MAX_SIZE[1] / max(height, 1))
    return max(1, int(width * f)), max(1, int(height * f))


class View3D(QWidget):
    def __init__(self, model, data, lookat_height: float, parent=None):
        super().__init__(parent)
        # The offscreen buffer of MuJoCo is sized by the model: large enough for any image of the view
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, MAX_SIZE[0])
        model.vis.global_.offheight = max(model.vis.global_.offheight, MAX_SIZE[1])
        self.model, self.data = model, data
        self.camera = Camera(lookat_height)
        self.renderer = None
        self.pixels = None             # last image (height, width, 3), kept alive for self.image
        self.image = None
        self._drag = None              # last mouse position of a drag
        self._resize_timer = QTimer(self, singleShot=True, interval=RESIZE_DELAY)
        self._resize_timer.timeout.connect(self._create_renderer)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

    # --- Rendering ---------------------------------------------------------------------------
    def _physical_size(self) -> tuple:
        ratio = self.devicePixelRatioF()
        return image_size(round(self.width() * ratio), round(self.height() * ratio))

    def _create_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        width, height = self._physical_size()
        self.renderer = mujoco.Renderer(self.model, height, width)

    def render_frame(self) -> None:
        """Renders the scene as it is now, then asks Qt to draw it."""
        if self.renderer is None:
            self._create_renderer()
        self.renderer.update_scene(self.data, camera=self.camera.mujoco())
        self.pixels = np.ascontiguousarray(self.renderer.render())
        height, width, _ = self.pixels.shape
        self.image = QImage(self.pixels.data, width, height, 3 * width, QImage.Format_RGB888)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self.image is None:
            painter.fillRect(self.rect(), Qt.black)
        else:
            painter.drawImage(QRect(0, 0, self.width(), self.height()), self.image)
        painter.end()

    def resizeEvent(self, event) -> None:
        if self.renderer is not None:
            self._resize_timer.start()        # restarted at every change of size
        super().resizeEvent(event)

    def close_renderer(self) -> None:
        self._resize_timer.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # --- Mouse -------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            p: QPointF = event.position()
            self.camera.turn(p.x() - self._drag.x(), p.y() - self._drag.y())
            self._drag = p

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = None

    def wheelEvent(self, event) -> None:
        self.camera.zoom(event.angleDelta().y() / 120)       # 120 per notch (Qt)
