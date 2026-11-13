"""The software (app/), without a screen: window, buttons, brain, language, pause, frames in step with real time.

Skipped if PySide6 is not installed (pip install -r requirements-app.txt). Qt draws off screen; the 3D view is
rendered by MuJoCo, like the virtual camera.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # before the application is created

from PySide6.QtWidgets import QApplication

from app.i18n import Texts
from app.window import MainWindow
from controller.policy import PolicyController

FRAME = 1 / 60


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


class FakeClock:
    """Real time of the frames, advanced by the test."""

    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def window(application):
    clock = FakeClock()
    w = MainWindow(Texts("en"), clock=clock)
    w.timer.stop()                         # the frames are played by the test
    w.resize(800, 500)
    w.show()
    application.processEvents()
    w.fake_clock = clock
    yield w
    w.close()


def play(window, frames: int) -> None:
    for _ in range(frames):
        window.fake_clock.now += FRAME
        window.tick()


def test_frames_in_step_with_real_time(window):
    window.tick()                          # starts the clock
    play(window, 60)                       # 1 s of frames
    s = window.session
    assert abs(s.t - 1.0) < s.dt
    height, width, _ = window.view.pixels.shape
    assert (width, height) == window.view._physical_size()
    assert window.view.pixels.std() > 10   # an image, not a flat color


def test_buttons_act_on_the_session(window):
    window.tick()
    window.b_throw.click()
    play(window, 10)
    assert window.session.attempt["action"] == "throw" and window.session.attempt["brain"] == "pid"
    assert window.state_text() == "Throw (PID) in progress…"
    window.b_drop.click()
    assert window.session.summary[-1]["result"] == "interrupted"
    assert window.session.attempt["action"] == "drop"
    window.b_center.click()
    play(window, 5)
    assert window.session.attempt["action"] == "center"


def test_brain_changed(window):
    window.tick()
    window.b_throw.click()
    play(window, 5)
    window.brain_buttons["imitation"].click()
    assert isinstance(window.session.sim.brain, PolicyController)
    assert window.session.summary[-1]["brain"] == "pid" and window.session.attempt is None
    assert window.state_text().startswith("Last attempt: Throw (PID), interrupted after ")
    window.b_throw.click()
    assert window.state_text() == "Throw (AI) in progress…"


def test_language(window):
    window.c_language.setCurrentIndex(window.c_language.findData("fr"))
    assert window.b_throw.text() == "Lancer" and window.windowTitle() == "ball-plate-sim : robot équilibriste"
    window.b_throw.click()
    assert window.state_text() == "Lancer (PID) en cours…"
    window.tick()
    play(window, 6)
    assert "0,1 s" in window.l_clock.text()


def test_pause(window):
    window.tick()
    play(window, 3)
    window.b_pause.click()
    assert window.b_pause.text() == "Resume"
    t = window.session.t
    play(window, 10)
    assert window.session.t == t
    window.b_pause.click()
    play(window, 10)
    assert window.session.t > t


def test_close(window):
    window.tick()
    window.b_throw.click()
    window.close()
    assert window.view.renderer is None and window.session.attempt is None
