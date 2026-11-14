"""Cost of a frame of the software on this computer: python -m app.benchmark

A frame does three things: the simulation catches up with real time (1/60 s of simulated time at 60 frames per
second), MuJoCo renders the 3D view and reads the image back, Qt draws the image in the window. This measures each of
them, for several sizes of the 3D view and several qualities of rendering, and tells how many frames per second they
allow together. Measured on the computer where it runs: the numbers depend on its graphics card.

Qualities: that of the software (app/scene.py), then one change at a time (antialiasing, size of the shadow map,
shadows, fog), and the look of phase D1 (the physics model as it is, without the display scene).
"""
import argparse
import platform
import time

import mujoco
import numpy as np

from app.scene import ANTIALIASING, MAX_SIZE, SHADOW_MAP, DisplayScene
from app.theme import scene_style
from app.view3d import Camera
from simulation.mujoco_physics import model_xml
from simulation.session import Session, sandbox_config

SIZES = ((1280, 720), (1920, 1080), (2560, 1440))     # (width, height) of the 3D view, in physical pixels
QUALITIES = {                                        # name: (antialiasing samples, shadow map, shadows, fog)
    "software": (ANTIALIASING, SHADOW_MAP, True, True),
    "antialiasing 8": (8, SHADOW_MAP, True, True),
    "shadows 2048": (ANTIALIASING, 2048, True, True),
    "no shadows": (ANTIALIASING, SHADOW_MAP, False, True),
    "no fog": (ANTIALIASING, SHADOW_MAP, True, False),
    "look of D1": None,
}
FRAME = 1 / 60                                       # (s) of simulated time per frame
_application = None                                  # the Qt application, if this module had to create it


def session_after_throw() -> Session:
    """A session with the ball rolling on the plate: the scene that is drawn most of the time."""
    s = Session(sandbox_config(), record=False)
    s.throw(0.10, 180, speed=0.4)
    for _ in range(round(0.5 / s.dt)):
        s.step()
    return s


def measure_simulation(frames: int) -> float:
    """(ms) to simulate FRAME s, the share of the simulation in a frame."""
    s = session_after_throw()
    start = time.perf_counter()
    for _ in range(frames):
        target = s.t + FRAME
        while s.t < target:
            s.step()
    return 1000 * (time.perf_counter() - start) / frames


def _model(session: Session, quality):
    """Model and data to render, and the function that keeps them in step with the physics."""
    if quality is None:                              # phase D1: the physics model itself
        model = mujoco.MjModel.from_xml_string(model_xml(session.cfg))
        model.vis.global_.offwidth, model.vis.global_.offheight = MAX_SIZE
        data = mujoco.MjData(model)

        def follow():
            data.qpos[:] = session.phys.data.qpos
            mujoco.mj_forward(model, data)
        return model, data, follow
    samples, shadow_map, _, _ = quality
    scene = DisplayScene(session.cfg, scene_style("light"))
    scene.model.vis.quality.offsamples = samples
    scene.model.vis.quality.shadowsize = shadow_map
    return scene.model, scene.data, lambda: scene.follow(session.phys.data)


def measure_rendering(quality, size: tuple, frames: int) -> float:
    """(ms) to render the 3D view of size (width, height) and read the image back, as the software does."""
    s = session_after_throw()
    model, data, follow = _model(s, quality)
    renderer = mujoco.Renderer(model, size[1], size[0])
    camera = Camera(s.cfg.geometry.height - 0.035).mujoco()

    def frame():
        follow()
        renderer.update_scene(data, camera=camera)
        if quality is not None:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = quality[2]
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = quality[3]
        return np.ascontiguousarray(renderer.render())

    for _ in range(3):                               # the first frames prepare the graphics card
        frame()
    start = time.perf_counter()
    for _ in range(frames):
        frame()
    elapsed = time.perf_counter() - start
    renderer.close()
    return 1000 * elapsed / frames


def measure_drawing(size: tuple, frames: int) -> float:
    """(ms) for Qt to turn the image into a QImage and draw it at the same size, as the 3D view does."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    global _application
    if QGuiApplication.instance() is None:           # Qt needs an application, kept alive as long as the module
        _application = QGuiApplication([])
    width, height = size
    pixels = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    target = QImage(width, height, QImage.Format_RGB32)
    start = time.perf_counter()
    for _ in range(frames):
        image = QImage(pixels.data, width, height, 3 * width, QImage.Format_RGB888)
        painter = QPainter(target)
        painter.drawImage(QRect(0, 0, width, height), image)
        painter.end()
    return 1000 * (time.perf_counter() - start) / frames


def graphics_card() -> str:
    """Name of the graphics card that renders, as OpenGL gives it (empty if it cannot be read)."""
    try:
        from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString
        model = mujoco.MjModel.from_xml_string("<mujoco/>")
        renderer = mujoco.Renderer(model, 8, 8)
        renderer._gl_context.make_current()
        text = f"{glGetString(GL_RENDERER).decode()} (OpenGL {glGetString(GL_VERSION).decode()})"
        renderer.close()
        return text
    except Exception:          # PyOpenGL or the context unavailable: the measurements stay valid without it
        return ""


def run(sizes=SIZES, frames: int = 30) -> dict:
    """All the measurements, in ms per frame: {"simulation": x, "drawing": {size: x}, "rendering": {quality:
    {size: x}}}."""
    return dict(simulation=measure_simulation(2 * frames),
                drawing={size: measure_drawing(size, frames) for size in sizes},
                rendering={name: {size: measure_rendering(q, size, frames) for size in sizes}
                           for name, q in QUALITIES.items()})


def report(results: dict, sizes) -> str:
    """The measurements as a table, then the frames per second they allow (1000 / total of the three shares)."""
    heads = "".join(f"{w}×{h}".rjust(13) for w, h in sizes)
    lines = [f"Simulation (1/60 s simulated per frame): {results['simulation']:.1f} ms",
             "", f"{'Drawing in the window (Qt), ms':34}{heads}",
             f"{'':34}" + "".join(f"{results['drawing'][size]:13.1f}" for size in sizes),
             "", f"{'3D rendering + read back, ms':34}{heads}"]
    for name, row in results["rendering"].items():
        lines.append(f"{name:34}" + "".join(f"{row[size]:13.1f}" for size in sizes))
    lines += ["", f"{'Frames per second allowed':34}{heads}"]
    for name, row in results["rendering"].items():
        total = [results["simulation"] + row[size] + results["drawing"][size] for size in sizes]
        lines.append(f"{name:34}" + "".join(f"{1000 / t:13.0f}" for t in total))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=30, help="frames measured per case (default 30)")
    args = parser.parse_args()
    print(f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}")
    card = graphics_card()
    if card:
        print(f"Graphics: {card}")
    print("Measuring (about a minute)...\n")
    print(report(run(SIZES, args.frames), SIZES))


if __name__ == "__main__":
    main()
