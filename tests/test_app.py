"""The software (app/), without a screen: window, buttons, brain, measurements, language, theme, pause, frames in
step with real time, measurement of the cost of a frame (app/benchmark.py).

Skipped if PySide6 is not installed (pip install -r requirements-app.txt). Qt draws off screen; the 3D view is
rendered by MuJoCo, like the virtual camera.
"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # before the application is created

from PySide6.QtWidgets import QApplication

from app.i18n import Texts
from app.theme import COLORS
from app.window import READOUT_EVERY, MainWindow
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
    w = MainWindow(Texts("en"), mode="light", clock=clock)
    w.timer.stop()                         # the frames are played by the test
    w.resize(900, 560)
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
    assert s.t == pytest.approx(1.0, abs=1.01 * s.dt)
    height, width, _ = window.view.pixels.shape
    assert (width, height) == window.view._physical_size()
    assert window.view.pixels.std() > 10   # an image, not a flat color


def test_slow_frames_caught_up(window):
    window.tick()
    window.fake_clock.now += 0.09          # a slow frame (90 ms): the simulation stays in step with real time
    window.tick()
    assert window.session.t == pytest.approx(0.09, abs=1.01 * window.session.dt)    # whole steps: one beyond at most


def test_buttons_act_on_the_session(window):
    window.tick()
    window.b_throw.click()
    play(window, 10)
    assert window.session.attempt["action"] == "throw" and window.session.attempt["brain"] == "pid"
    assert window.state_text() == "Throw (PID) in progress"
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
    window.s_brain.buttons["imitation"].click()
    assert isinstance(window.session.sim.brain, PolicyController)
    assert window.session.summary[-1]["brain"] == "pid" and window.session.attempt is None
    assert window.last_attempt_text().startswith("Throw · PID · interrupted after ")
    window.b_throw.click()
    assert window.state_text() == "Throw (AI) in progress"


def test_measurements(window):
    window.tick()
    window.b_center.click()
    play(window, 2 * READOUT_EVERY)
    distance = window.readouts["distance_mm"].value.text()
    assert distance != "—" and float(distance) < 5
    assert window.readouts["tilt_x_deg"].value.text().endswith("°")
    assert window.view.overlay.bottom_left.startswith("Ball: x ")
    window.session.throw(0.10, 180, speed=1.0)            # too fast: falls
    for _ in range(round(3.0 / window.session.dt)):
        window.session.step()
    window.refresh()
    assert window.readouts["distance_mm"].value.text() == "—" and window.view.overlay.bottom_left == ""
    assert window.state_text() == "The ball has fallen"
    assert window.view.overlay.state_color == COLORS["light"]["bad"]


def test_language(window):
    window.s_language.buttons["fr"].click()
    assert window.b_throw.text() == "Lancer" and window.windowTitle() == "ball-plate-sim : robot équilibriste"
    assert window.readouts["speed_mm_s"].name.text() == "VITESSE"
    window.b_throw.click()
    assert window.state_text() == "Lancer (PID) en cours"
    window.tick()
    play(window, 6)
    assert window.view.overlay.bottom_right.startswith("t = 0,1 s")


def test_theme(window):
    window.tick()
    play(window, 2)
    light = window.view.scene
    window.s_mode.buttons["dark"].click()
    assert window.mode == "dark" and window.view.scene is not light and window.view.renderer is None
    assert COLORS["dark"]["bg"] in window.styleSheet() and window.view.overlay.colors is COLORS["dark"]
    play(window, 2)                        # the new scene is rendered
    assert window.view.pixels is not None and window.view.renderer is not None


def test_pause(window):
    window.tick()
    play(window, 3)
    window.b_pause.click()
    assert window.b_pause.text() == "Resume" and window.view.overlay.state == "Paused"
    t = window.session.t
    play(window, 10)
    assert window.session.t == t
    window.b_pause.click()
    play(window, 10)
    assert window.session.t > t


def test_benchmark(application):
    from app.benchmark import QUALITIES, report, run
    sizes = ((64, 48),)
    results = run(sizes, frames=2)
    assert results["simulation"] > 0 and results["drawing"][sizes[0]] > 0
    assert set(results["rendering"]) == set(QUALITIES)
    assert all(row[sizes[0]] > 0 for row in results["rendering"].values())
    text = report(results, sizes)
    assert "64×48" in text and "look of D1" in text


def test_close(window):
    window.tick()
    window.b_throw.click()
    window.close()
    assert window.view.renderer is None and window.session.attempt is None
