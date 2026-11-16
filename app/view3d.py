"""The robot in 3D: the display scene (app/scene.py) rendered off screen by MuJoCo, drawn in a Qt widget with rounded
corners, with what is drawn over it: texts (Overlay), the setpoint and its path, the arrow of a gesture, the x and y
axes of the plate.

Mouse: with no tool, left drag turns the view around the plate; the wheel zooms (then turned(dx, dy) or
zoomed(notches): another view can follow). With a tool (tool = "throw",
"drop" or "push"), a press on the plate starts the gesture (app/gestures.py), a press elsewhere turns the view:
- throw: drag from where the ball starts; on release, thrown(x, y, vx, vy) (frame of the plate);
- drop: click where the ball falls, or drag from there to give it a horizontal speed; on release, dropped(x, y,
  vx, vy) (frame of the plate; no speed for a click); while dragging, its fall from drop_height is drawn;
- push: drag from the ball; on release, pushed(Δvx, Δvy) (horizontal, frame of the world);
- draw: drag a path on the plate (a point every DRAW_STEP); on release, drawn(points) (frame of the plate). It is
  drawn in the good color while it stays within draw_radius of the center, in the bad color beyond.
The image has the size of the widget in physical pixels (high-resolution screens), reduced if needed to max_pixels
pixels (the window shares RENDER_PIXELS between its views) and to scene.MAX_SIZE; Qt then enlarges it to the widget.
The cost of a frame grows with the number of pixels rendered, not with the parts of the robot (measured with
python -m app.benchmark on an integrated card: 25 ms at 1920 × 1080, 39 ms at 2560 × 1440, as much as the bare physics
model). After a change of size, the renderer is created again at the new size once the size has stopped changing for
RESIZE_DELAY; until then, the last image is stretched.
"""
from dataclasses import dataclass, field
import math

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from app.gestures import MIN_SPEED, drop_flight, plate_hit, velocity
from app.scene import MAX_SIZE, VIEW_DISTANCE, DisplayScene

RESIZE_DELAY = 150              # (ms)
RENDER_PIXELS = 1920 * 1080     # at most this many pixels rendered for the 3D, all views together
DEGREES_PER_PIXEL = 0.3         # turn of the view per pixel of mouse drag
ZOOM_PER_NOTCH = 0.9            # distance multiplied by this per notch of the wheel (forward: closer)
DISTANCE = (0.25, 2.0)          # (m) closest and farthest view
ELEVATION = (-89.0, -5.0)       # (°) from straight above to almost level with the plate
RADIUS = 8                      # (pixels) corners of the view
AXIS_LENGTH = 28                # (pixels) axes of the plate, drawn in the corner
PUSH_REACH = 3                  # a push starts within this many ball radii of the ball
DRAW_STEP = 0.002               # (m) between the points of a drawn path
FIT_ASPECT = 1.0                # a view narrower than this (width / height; side by side) moves its camera back,
                                # to keep the width of the scene in sight (the default window is wider: 1.2)


class Camera:
    """Point of view around the plate: azimuth and elevation (°), distance (m), point looked at, vertical field of
    view (°). Computes, as MuJoCo does for its free camera, the pixel of a point and the ray of a pixel. Without Qt."""

    def __init__(self, lookat_height: float, fovy: float = 45.0):
        self.azimuth, self.elevation, self.distance = 90.0, -27.0, VIEW_DISTANCE
        self.fit = 1.0                  # the camera is that many times farther than `distance` (narrow view)
        self.lookat = (0.0, 0.0, lookat_height)
        self.fovy = fovy

    def turn(self, dx: float, dy: float) -> None:
        """Mouse drag of (dx, dy) pixels: to the right, the scene turns to the right; downward, the scene is
        seen from higher up."""
        self.azimuth = (self.azimuth - DEGREES_PER_PIXEL * dx) % 360.0
        self.elevation = min(max(self.elevation - DEGREES_PER_PIXEL * dy, ELEVATION[0]), ELEVATION[1])

    def zoom(self, notches: float) -> None:
        """notches > 0: closer."""
        self.distance = min(max(self.distance * ZOOM_PER_NOTCH ** notches, DISTANCE[0]), DISTANCE[1])

    def glide(self, lookat, distance: float, dt: float, time_constant: float) -> None:
        """Moves the point looked at and the distance toward (lookat, distance), smoothly: the gap shrinks by
        exp(-dt / time_constant) during dt."""
        k = 1.0 - math.exp(-dt / time_constant)
        self.lookat = tuple(a + k * (b - a) for a, b in zip(self.lookat, lookat))
        self.distance += k * (distance - self.distance)

    def mujoco(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = self.azimuth, self.elevation, self.fit * self.distance
        cam.lookat[:] = self.lookat
        return cam

    def frame(self) -> tuple:
        """Eye and axes (forward f, up u, right r = f × u) of the camera: f = (cos e·cos a, cos e·sin a, sin e),
        u = (-sin e·cos a, -sin e·sin a, cos e), the eye at fit·distance behind the point looked at."""
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        f = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        u = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
        return np.array(self.lookat) - self.fit * self.distance * f, f, u, np.cross(f, u)

    def project(self, point, width: float, height: float):
        """Pixel (x, y) of a point of the world in a view of width × height, or None if it is behind the eye."""
        eye, f, u, r = self.frame()
        v = np.asarray(point, dtype=float) - eye
        depth = float(v @ f)
        if depth <= 1e-6:
            return None
        t = math.tan(math.radians(self.fovy) / 2)
        nx, ny = float(v @ r) / (depth * t * width / height), float(v @ u) / (depth * t)
        return (nx + 1) / 2 * width, (1 - ny) / 2 * height

    def ray(self, x: float, y: float, width: float, height: float) -> tuple:
        """(eye, direction) of the ray through the pixel (x, y) of a view of width × height."""
        eye, f, u, r = self.frame()
        t = math.tan(math.radians(self.fovy) / 2)
        nx, ny = 2 * x / width - 1, 1 - 2 * y / height
        return eye, f + nx * t * width / height * r + ny * t * u

    def screen_axes(self) -> tuple:
        """Directions on the screen (right, up) of the x and y axes of the world: ((xr, xu), (yr, yu))."""
        _, _, u, r = self.frame()
        return (float(r[0]), float(u[0])), (float(r[1]), float(u[1]))


def image_size(width: int, height: int, max_pixels: int = RENDER_PIXELS) -> tuple:
    """Size of the image rendered for a widget of width × height physical pixels: the same, reduced if needed (same
    proportions) to max_pixels pixels and to fit in MAX_SIZE, at least 1 × 1."""
    f = min(1.0, math.sqrt(max_pixels / max(width * height, 1)),
            MAX_SIZE[0] / max(width, 1), MAX_SIZE[1] / max(height, 1))
    return max(1, int(width * f + 1e-9)), max(1, int(height * f + 1e-9))


@dataclass
class Overlay:
    """What is drawn over the 3D view, and the colors of the theme (keys of app/theme.py)."""
    state: str = ""
    state_color: str = "#16A34A"
    hint: str = ""
    bottom_left: str = ""
    bottom_right: str = ""
    setpoint: tuple = None          # (x, y) in the frame of the plate, or None
    path: list = field(default_factory=list)      # path of the setpoint (frame of the plate)
    colors: dict = field(default_factory=dict)


class View3D(QWidget):
    thrown = Signal(float, float, float, float)
    dropped = Signal(float, float, float, float)
    pushed = Signal(float, float)
    turned = Signal(float, float)       # the view was turned by the mouse, by (dx, dy) pixels
    zoomed = Signal(float)              # the view was zoomed by the wheel, by this many notches
    drawn = Signal(list)                # a path drawn on the plate: [(x, y)]

    def __init__(self, scene: DisplayScene, physics_data, lookat_height: float, plate_radius: float,
                 ball_radius: float, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.physics_data = physics_data
        self.plate_radius, self.ball_radius = plate_radius, ball_radius
        self.camera = Camera(lookat_height, scene.model.vis.global_.fovy)
        self.overlay = Overlay()
        self.tool = None               # "throw", "drop", "push" or None
        self.gesture = None            # dict(kind, start, end) while a gesture is drawn
        self.hover = None              # point of the plate under the mouse (drop tool)
        self.drop_height = 0.0         # (m) height of a drop, for drawing its fall (set by the window)
        self.draw_radius = math.inf    # (m) a drawn path must stay within this distance of the center
        self.describe = lambda kind, start, end: ("", "accent")    # text and color of a gesture (set by the window)
        self.renderer = None
        self.max_pixels = RENDER_PIXELS     # pixels rendered at most (set by the window: shared between the views)
        self.pixels = None             # last image (height, width, 3), kept alive for self.image
        self.image = None
        self._drag = None              # last mouse position while turning the view
        self._resize_timer = QTimer(self, singleShot=True, interval=RESIZE_DELAY)
        self._resize_timer.timeout.connect(self._create_renderer)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def set_scene(self, scene: DisplayScene) -> None:
        """Another display scene (theme changed): the renderer is created again for it."""
        self.close_renderer()
        self.scene = scene

    def set_max_pixels(self, max_pixels: int) -> None:
        """At most this many pixels rendered: the renderer is created again if its size changes."""
        if max_pixels != self.max_pixels:
            self.max_pixels = max_pixels
            if self.renderer is not None:
                self._resize_timer.start()

    def set_tool(self, tool) -> None:
        self.tool, self.gesture, self.hover = tool, None, None
        self.setCursor(Qt.ArrowCursor if tool is None else Qt.CrossCursor)
        self.update()

    # --- Rendering ---------------------------------------------------------------------------
    def _physical_size(self) -> tuple:
        ratio = self.devicePixelRatioF()
        return image_size(round(self.width() * ratio), round(self.height() * ratio), self.max_pixels)

    def _create_renderer(self) -> None:
        self._free_renderer()
        width, height = self._physical_size()
        self.renderer = mujoco.Renderer(self.scene.model, height, width)

    def _free_renderer(self) -> None:
        """MuJoCo frees the objects of a renderer in the OpenGL context that is current: its own must be, otherwise
        those of another view (side by side) would be freed in its place (measured: that view then draws garbage)."""
        if self.renderer is not None:
            if self.renderer._gl_context:
                self.renderer._gl_context.make_current()
            self.renderer.close()
            self.renderer = None

    def render_frame(self) -> None:
        """Renders the robot as it is now in the physics, then asks Qt to draw it."""
        if self.renderer is None:
            self._create_renderer()
        self._fit()
        self.scene.view(self.renderer._mjr_context, self.camera.lookat, self.camera.fit * self.camera.distance)
        self.scene.follow(self.physics_data)
        self.renderer.update_scene(self.scene.data, camera=self.camera.mujoco())
        self.scene.prepare(self.renderer.scene)                        # fog; the floor casts no shadow
        self.pixels = np.ascontiguousarray(self.renderer.render())
        height, width, _ = self.pixels.shape
        self.image = QImage(self.pixels.data, width, height, 3 * width, QImage.Format_RGB888)
        self.update()

    def close_renderer(self) -> None:
        self._resize_timer.stop()
        self._free_renderer()

    def resizeEvent(self, event) -> None:
        if self.renderer is not None:
            self._resize_timer.start()        # restarted at every change of size
        super().resizeEvent(event)

    # --- Between the screen and the plate ----------------------------------------------------
    def _plate(self) -> tuple:
        """Origin (on the surface) and rotation of the plate, as displayed."""
        b = self.scene.data.body("plate")
        return b.xpos.copy(), b.xmat.reshape(3, 3).copy()

    def _fit(self) -> None:
        """A view narrower than FIT_ASPECT: the camera moves back, to see the same width of the scene."""
        self.camera.fit = max(1.0, FIT_ASPECT * self.height() / max(self.width(), 1))

    def to_plate(self, x: float, y: float):
        """Point (x, y) of the plane of the plate (frame of the plate) under the pixel (x, y), or None."""
        self._fit()
        eye, direction = self.camera.ray(x, y, self.width(), self.height())
        return plate_hit(eye, direction, *self._plate())

    def to_screen(self, point):
        """Pixel of a point (x, y) of the plate surface, or (x, y, z) at z above it (frame of the plate), or None."""
        self._fit()
        position, R = self._plate()
        local = np.array([point[0], point[1], point[2] if len(point) > 2 else 0.0])
        return self.camera.project(position + R @ local, self.width(), self.height())

    def ball_on_plate(self) -> tuple:
        """Position (x, y) of the ball in the frame of the plate."""
        position, R = self._plate()
        local = R.T @ (self.scene.data.body("ball").xpos - position)
        return float(local[0]), float(local[1])

    # --- Gestures (called by the mouse; by the tests directly) -------------------------------
    def press(self, x: float, y: float) -> bool:
        """A press of the mouse at the pixel (x, y): starts the gesture of the tool, or returns False (the view
        turns instead)."""
        if self.tool is None:
            return False
        p = self.to_plate(x, y)
        if p is None or math.hypot(*p) > self.plate_radius - self.ball_radius:
            return False
        start = p
        if self.tool == "push":
            start = self.ball_on_plate()
            if math.hypot(p[0] - start[0], p[1] - start[1]) > PUSH_REACH * self.ball_radius:
                return False
        self.gesture, self.hover = dict(kind=self.tool, start=start, end=start, points=[start]), None
        return True

    def move(self, x: float, y: float) -> None:
        p = self.to_plate(x, y)
        if self.gesture is not None and p is not None:
            self.gesture["end"] = p
            last = self.gesture["points"][-1]
            if self.gesture["kind"] == "draw" and math.hypot(p[0] - last[0], p[1] - last[1]) >= DRAW_STEP:
                self.gesture["points"].append(p)
        elif self.tool == "drop":
            self.hover = p if p is not None and math.hypot(*p) <= self.plate_radius - self.ball_radius else None
        self.update()

    def release(self) -> None:
        g, self.gesture = self.gesture, None
        self.update()
        if g is None:
            return
        if g["kind"] == "draw":
            self.drawn.emit(g["points"])
            return
        vx, vy = velocity(g["start"], g["end"])
        moving = math.hypot(vx, vy) >= MIN_SPEED
        if g["kind"] == "drop":
            self.dropped.emit(g["start"][0], g["start"][1], *((vx, vy) if moving else (0.0, 0.0)))
        elif not moving:
            return
        elif g["kind"] == "throw":
            self.thrown.emit(g["start"][0], g["start"][1], vx, vy)
        else:
            dv = self._plate()[1] @ np.array([vx, vy, 0.0])       # frame of the plate -> world, horizontal part
            self.pushed.emit(float(dv[0]), float(dv[1]))

    # --- Mouse -------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not self.press(event.position().x(), event.position().y()):
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:
        q = event.position()
        if self._drag is not None:
            dx, dy = q.x() - self._drag.x(), q.y() - self._drag.y()
            self.camera.turn(dx, dy)
            self._drag = q
            self.turned.emit(dx, dy)
        elif self.tool is not None:
            self.move(q.x(), q.y())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = None
            self.release()

    def leaveEvent(self, event) -> None:
        self.hover = None
        self.update()

    def wheelEvent(self, event) -> None:
        notches = event.angleDelta().y() / 120               # 120 per notch (Qt)
        self.camera.zoom(notches)
        self.zoomed.emit(notches)

    # --- Drawing -----------------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        o, c = self.overlay, self.overlay.colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)     # an image smaller than the widget, enlarged smoothly
        rect = QRectF(self.rect())
        frame = QPainterPath()
        frame.addRoundedRect(rect, RADIUS, RADIUS)
        p.setClipPath(frame)
        if self.image is None:
            p.fillRect(rect, QColor(c.get("bg", "#000000")))
        else:
            p.drawImage(rect, self.image)
        if c:
            self._setpoint(p, c)
            self._gesture(p, c)
        p.setClipping(False)
        if c:
            p.setPen(QPen(QColor(c["border"]), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
            chip = self._chip(p, o.state, o.state_color, c)
            p.setFont(self._font(12))
            p.setPen(QColor(c["text2"]))
            text = rect.adjusted(16, 0, -16, -12)
            if chip.right() + 16 + p.fontMetrics().horizontalAdvance(o.hint) <= rect.width() - 16:
                p.drawText(rect.adjusted(16, 22, -16, 0), Qt.AlignTop | Qt.AlignRight, o.hint)
            else:                                           # a narrow view: under the chip
                p.drawText(QRectF(16, chip.bottom() + 8, rect.width() - 32, 64),
                           Qt.AlignTop | Qt.AlignRight | Qt.TextWordWrap, o.hint)
            p.drawText(text, Qt.AlignBottom | Qt.AlignLeft, o.bottom_left)
            metrics = p.fontMetrics()
            if metrics.horizontalAdvance(o.bottom_left) + 24 + metrics.horizontalAdvance(o.bottom_right) > text.width():
                text = text.adjusted(0, 0, 0, -metrics.height() - 2)          # a narrow view: one line higher
            p.drawText(text, Qt.AlignBottom | Qt.AlignRight, o.bottom_right)
            self._axes(p, QPointF(rect.width() - 44, rect.height() - 64), c)
        p.end()

    def _font(self, size: int, bold: bool = False) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(size)
        if bold:
            f.setWeight(QFont.DemiBold)
        return f

    def _point(self, point):
        q = self.to_screen(point)
        return None if q is None else QPointF(*q)

    def _setpoint(self, p: QPainter, c: dict) -> None:
        """The path of the setpoint, thin, and the setpoint now, a ring."""
        color = QColor(c["accent"])
        points = [q for q in (self._point(xy) for xy in self.overlay.path) if q is not None]
        if len(points) > 1:
            faint = QColor(color)
            faint.setAlpha(110)
            p.setPen(QPen(faint, 1.5, Qt.DashLine))
            path = QPainterPath(points[0])
            for q in points[1:]:
                path.lineTo(q)
            p.drawPath(path)
        if self.overlay.setpoint is not None:
            q = self._point(self.overlay.setpoint)
            if q is not None:
                p.setPen(QPen(color, 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(q, 7, 7)
                p.drawPoint(q)
        if self.hover is not None:
            q = self._point(self.hover)
            if q is not None:
                p.setPen(QPen(color, 2, Qt.DotLine))
                p.drawEllipse(q, 9, 9)

    def _gesture(self, p: QPainter, c: dict) -> None:
        """The arrow of the gesture being drawn, with its speed (and the physical limit of a throw); for a drop,
        its fall."""
        g = self.gesture
        if g is None:
            return
        if g["kind"] == "draw":
            self._path(p, g["points"], c)
            return
        a, b = self._point(g["start"]), self._point(g["end"])
        if a is None or b is None:
            return
        text, color_key = self.describe(g["kind"], g["start"], g["end"])
        color = QColor(c[color_key])
        if g["kind"] == "drop":
            self._fall(p, g, a, color)
        p.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(a, b)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        n = math.hypot(dx, dy)
        if n > 1:
            ux, uy = dx / n, dy / n
            for side in (1, -1):        # head of the arrow: two strokes at ±30°
                hx = ux * math.cos(math.radians(150)) - side * uy * math.sin(math.radians(150))
                hy = uy * math.cos(math.radians(150)) + side * ux * math.sin(math.radians(150))
                p.drawLine(b, b + QPointF(12 * hx, 12 * hy))
        p.setBrush(color)
        p.drawEllipse(a, 4, 4)
        if text:
            p.setFont(self._font(13, bold=True))
            box = QRectF(b.x() + 12, b.y() - 30, p.fontMetrics().horizontalAdvance(text) + 16, 24)
            background = QColor(c["surface"])
            background.setAlpha(235)
            p.setPen(QPen(color, 1))
            p.setBrush(background)
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(c["text"]))
            p.drawText(box, Qt.AlignCenter, text)

    def _path(self, p: QPainter, points: list, c: dict) -> None:
        """A path being drawn, and the straight line that will close it (dashed)."""
        inside = all(math.hypot(*q) <= self.draw_radius for q in points)
        color = QColor(c["good" if inside else "bad"])
        screen = [q for q in (self._point(q) for q in points) if q is not None]
        if len(screen) < 2:
            return
        line = QPainterPath(screen[0])
        for q in screen[1:]:
            line.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(line)
        p.setPen(QPen(color, 1.5, Qt.DashLine))
        p.drawLine(screen[-1], screen[0])

    def _fall(self, p: QPainter, g: dict, start: QPointF, color: QColor) -> None:
        """Fall of a drop (app/gestures.drop_flight, the plate as it is now): from the plate up to the point of
        release (dotted), the parabola of the bottom of the ball down to the plate (dashed), a ring where it lands."""
        vx, vy = velocity(g["start"], g["end"])
        if math.hypot(vx, vy) < MIN_SPEED:
            vx = vy = 0.0
        landing, _, flight = drop_flight(g["start"], vx, vy, self.drop_height, self._plate()[1])
        points = [q for q in (self._point(xyz) for xyz in flight) if q is not None]
        faint = QColor(color)
        faint.setAlpha(170)
        p.setBrush(Qt.NoBrush)
        if points:
            p.setPen(QPen(faint, 1.5, Qt.DotLine))
            p.drawLine(start, points[0])
        if len(points) > 1:
            p.setPen(QPen(faint, 2, Qt.DashLine))
            path = QPainterPath(points[0])
            for q in points[1:]:
                path.lineTo(q)
            p.drawPath(path)
        q = self._point(landing)
        if q is not None:
            p.setPen(QPen(color, 2))
            p.drawEllipse(q, 9, 9)

    def _chip(self, p: QPainter, text: str, dot: str, c: dict) -> QRectF:
        p.setFont(self._font(12, bold=True))
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
        return r

    def _axes(self, p: QPainter, origin: QPointF, c: dict) -> None:
        """The x and y axes of the plate as seen from the camera (they turn with the view)."""
        p.setFont(self._font(11, bold=True))
        for (right, up), name, color in zip(self.camera.screen_axes(), "xy", (c["axis_x"], c["axis_y"])):
            end = origin + QPointF(AXIS_LENGTH * right, -AXIS_LENGTH * up)
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(origin, end)
            p.drawText(origin + QPointF((AXIS_LENGTH + 9) * right - 4, -(AXIS_LENGTH + 9) * up + 4), name)
