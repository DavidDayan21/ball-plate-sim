"""Check of the software as packaged (release/sapir.spec): SAPIR.exe --self-check [FILE], or python -m app
--self-check [FILE]. What the package must hold is loaded and used once: the texts of every language, the fonts, the
two brains (each one brings back a ball dropped off center), the OpenGL library, the robot shown, the window (not
shown on the screen, with its own settings: those of the user are left untouched); then the 3D is drawn once. A
file left out of the package, the most common fault of a package, makes its check fail.

One line per check, in FILE (UTF-8) and on the output when there is one (SAPIR.exe has none), then the verdict:
    ok       texts: en, fr
    FAILED   brains: FileNotFoundError: ...
    skipped  drawing: no usable OpenGL on this computer (...)
Exit code: 0 all passed; 1 a check failed; 2 all passed but the drawing, impossible on this computer (OpenGL missing
or too old: no graphics card): everything else is still checked.
"""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys
import tempfile

import mujoco
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from app import NAME, VERSION
from app.i18n import DEFAULT, LANGUAGES, Texts
from app.readouts import readouts
from app.scene import DisplayScene
from app.theme import FONTS, load_fonts, scene_style
from app.window import BRAINS, MainWindow
from simulation.session import Session, sandbox_config

OPTION = "--self-check"
DROP = (0.05, 30.0)     # (m, °) the ball dropped there, at rest: every brain must bring it back to the center
SETTLE = 3.0            # (s) simulated (measured: 1.4 mm from the center for the PID brain, 1.5 mm for the AI brain)
NEAR = 5.0              # (mm) from the center, at the end
SIZE = (900, 560)       # (pixels) of the window
CONTRAST = 10.0         # standard deviation of the pixels of the drawing, at least (a black or uniform image: none)


class NoOpenGL(Exception):
    """This computer cannot draw the 3D (no usable OpenGL): not a fault of the package."""


def _no_opengl_errors() -> tuple:
    """The errors of a renderer that cannot be created: MuJoCo, no usable OpenGL context (Windows, Linux); Mac,
    no pixel format."""
    errors = [mujoco.FatalError]
    if sys.platform == "darwin":
        from mujoco.cgl.cgl import CGLError
        errors.append(CGLError)
    return tuple(errors)


def _texts(state) -> str:
    for code in LANGUAGES:
        Texts(code)("window.title", name=NAME)
    return ", ".join(LANGUAGES)


def _fonts(state) -> str:
    files = sorted(FONTS.glob("*.ttf"))
    if not files:
        raise FileNotFoundError(f"no font in {FONTS}")
    load_fonts()
    if "Inter" not in QFontDatabase.families():
        raise RuntimeError("Qt does not know the font Inter")
    return f"Inter, {len(files)} files"


def _brains(state) -> str:
    done = []
    for brain in BRAINS:
        s = Session(replace(sandbox_config(), brain=brain), record=False)
        s.throw(*DROP)
        while s.t < SETTLE - 1e-9:
            s.step()
        distance = readouts(s)["distance_mm"]
        if s.state == "fallen" or distance is None or distance > NEAR:
            raise RuntimeError(f"{brain}: the ball is not back to the center ({s.state}, {distance} mm)")
        done.append(f"{brain} {distance:.1f} mm")
    return f"ball dropped {100 * DROP[0]:g} cm off center, {SETTLE:g} s later: " + ", ".join(done)


def _opengl(state) -> str:
    from mujoco.rendering.classic.gl_context import GLContext       # its library: GLFW, or CGL on Mac
    return GLContext.__module__


def _robot(state) -> str:
    model = DisplayScene(sandbox_config(), scene_style("light")).model
    return f"{model.nmesh} meshes, {model.nmeshface} triangles"


def _window(state) -> str:
    state["folder"] = tempfile.mkdtemp()
    state["settings"] = settings = QSettings(os.path.join(state["folder"], "settings.ini"), QSettings.IniFormat)
    window = MainWindow(Texts(DEFAULT), mode="light", settings=settings)
    state["window"] = window
    window.timer.stop()
    window.setAttribute(Qt.WA_DontShowOnScreen)
    window.resize(*SIZE)
    window.show()
    QApplication.processEvents()
    return f"{window.width()} × {window.height()} pixels"


def _drawing(state) -> str:
    if "window" not in state:
        raise RuntimeError("no window")
    try:
        state["window"].tick()                  # one frame: the renderer is created, the 3D rendered
    except _no_opengl_errors() as error:
        raise NoOpenGL(f"{type(error).__name__}: {error}") from error
    pixels = state["window"].view.pixels
    contrast = float(np.std(pixels))
    if contrast < CONTRAST:
        raise RuntimeError(f"the image is uniform (standard deviation {contrast:.1f})")
    height, width, _ = pixels.shape
    return f"{width} × {height} pixels, standard deviation {contrast:.0f}"


CHECKS = (("texts", _texts), ("fonts", _fonts), ("brains", _brains), ("OpenGL", _opengl), ("robot shown", _robot),
          ("window", _window), ("drawing", _drawing))


def run(report=None) -> int:
    """All the checks, in order (a failed one does not stop the next ones); writes the report in the file `report`
    (None: none) and on the output. Returns the exit code."""
    QApplication.instance() or QApplication([])
    lines, failed, skipped, state = [], False, False, {}
    for name, check in CHECKS:
        try:
            lines.append(f"ok       {name}: {check(state)}")
        except NoOpenGL as error:
            skipped = True
            lines.append(f"skipped  {name}: no usable OpenGL on this computer ({error})")
        except Exception as error:
            failed = True
            lines.append(f"FAILED   {name}: {type(error).__name__}: {error}")
    if "window" in state:
        state["window"].close()                 # it then writes its geometry in its own settings
        state["settings"].sync()
    if "folder" in state:
        shutil.rmtree(state["folder"], ignore_errors=True)
    verdict = "failed" if failed else "passed, without the drawing" if skipped else "passed"
    lines.append(f"{NAME} {VERSION} self-check: {verdict}")
    text = "\n".join(lines) + "\n"
    if report is not None:
        Path(report).write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        print(text, end="")
    return 1 if failed else 2 if skipped else 0
