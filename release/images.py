"""The images of the README, made by the software itself, off screen: python -m release.images (about 2 minutes),
written in docs/images. Under Linux without a screen: QT_QPA_PLATFORM=offscreen and an X server (xvfb-run).
  side-by-side.png  the window, both brains side by side, a ball thrown to each (dark theme)
  robot.png         the robot alone, from three quarters, the ball at rest (light theme)
  catch.gif         a ball thrown and caught by the PID brain, in the 3D view (dark theme)
The time of the window is simulated (60 frames per simulated second): the images do not depend on the speed of the
computer, only on its OpenGL (the pixels of the 3D can differ slightly from one graphics card to another).
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.i18n import Texts
from app.scene import DisplayScene
from app.theme import scene_style
from app.view3d import RESIZE_DELAY, Camera
from app.window import MainWindow
from simulation.session import Session, sandbox_config

IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"
FRAME = 1 / 60                              # (s) of simulated time per frame of the window
THROW = (-0.08, -0.05, 0.45, 0.25)          # (m, m, m/s, m/s) start of the ball on the plate and its velocity
SETTLE = 1.5                                # (s) before the throw: the ball at rest, the frames per second measured
DISTANCE = 0.5                              # (m) of the camera in the window (the default view: 0.68)
ROBOT_VIEW = (125.0, -18.0, 0.46, 0.06)     # azimuth (°), elevation (°), distance (m), point looked at: m under
                                            # the plate


def _rgb(image: QImage) -> Image.Image:
    image = image.convertToFormat(QImage.Format_RGB888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return Image.fromarray(rows[:, :3 * image.width()].reshape(image.height(), image.width(), 3).copy())


class _Window:
    """The window, not shown on the screen, with a simulated clock."""

    def __init__(self, mode: str, size: tuple, split: bool = False):
        QApplication.instance() or QApplication([])
        self.now = 0.0
        self.window = MainWindow(Texts("en"), mode=mode, clock=lambda: self.now)
        self.window.timer.stop()
        self.window.setAttribute(Qt.WA_DontShowOnScreen)
        self.window.resize(*size)
        self.window.show()
        self.window.set_split(split)
        QTest.qWait(3 * RESIZE_DELAY)           # the views render at their final size
        for robot in self.window.robots:
            robot.view.camera.distance = DISTANCE
        self.advance(SETTLE)

    def advance(self, seconds: float) -> None:
        for _ in range(round(seconds / FRAME)):
            self.now += FRAME
            self.window.tick()
        self.window.refresh()                   # measurements and texts over the views, up to date
        QApplication.processEvents()

    def close(self) -> None:
        self.window.close()


def side_by_side(path, size=(1800, 820), after=0.5) -> None:
    """The window side by side, `after` s after a ball is thrown to both robots."""
    w = _Window("dark", size, split=True)
    w.window.throw(*THROW)
    w.advance(after)
    _rgb(w.window.grab().toImage()).save(path)
    w.close()


def robot(path, size=(1600, 900)) -> None:
    """The robot alone (not the window), the ball at rest at the center, the PID brain lit."""
    session = Session(sandbox_config(), record=False)
    for _ in range(round(0.5 / session.cfg.dt_physics)):
        session.step()
    scene = DisplayScene(session.cfg, scene_style("light"))
    scene.show("pid", session.ball, "en")
    azimuth, elevation, distance, below = ROBOT_VIEW
    camera = Camera(session.cfg.geometry.height - below)
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    renderer = mujoco.Renderer(scene.model, size[1], size[0])
    scene.view(renderer._mjr_context, camera.lookat, camera.distance)
    scene.follow(session.phys.data)
    renderer.update_scene(scene.data, camera=camera.mujoco())
    scene.prepare(renderer.scene)
    Image.fromarray(renderer.render().copy()).save(path)
    renderer.close()


def catch(path, size=(1220, 560), seconds=5.0, fps=20) -> None:
    """GIF of the 3D view of the PID brain: a ball thrown, then caught; `fps` images per simulated second (GIF counts
    in hundredths of a second: 100 / fps whole, for the GIF to play in real time)."""
    w = _Window("dark", size)
    w.window.throw(*THROW)
    frames = []
    for _ in range(round(seconds * fps)):
        w.advance(1 / fps)
        frames.append(_rgb(w.window.view.grab().toImage()))
    w.close()
    palette = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)   # one for all
    images = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                   optimize=True)


def main(folder=IMAGES) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    side_by_side(folder / "side-by-side.png")
    robot(folder / "robot.png")
    catch(folder / "catch.gif")


if __name__ == "__main__":
    main()
