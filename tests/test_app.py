"""The software (app/), without a screen: window, tools and gestures on the plate, setpoint, brain, measurements,
language, theme, pause, frames in step with real time, measurement of the cost of a frame (app/benchmark.py).

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


def test_tools(window):
    buttons = window.tool_buttons
    buttons["throw"].click()
    assert window.view.tool == "throw" and buttons["throw"].isChecked()
    buttons["drop"].click()                    # one tool at a time
    assert window.view.tool == "drop" and not buttons["throw"].isChecked() and window.height_row.isVisibleTo(window)
    buttons["drop"].click()                    # clicked again: no tool, the mouse turns the view
    assert window.view.tool is None and not buttons["drop"].isChecked()
    assert window.view.overlay.hint == "Drag: turn · Wheel: zoom"


def test_throw_with_the_mouse(window):
    window.tick()                              # the plate is drawn: the screen and the plate can be related
    window.tool_buttons["throw"].click()
    view = window.view
    a, b = view.to_screen((-0.08, 0.0)), view.to_screen((-0.04, 0.0))
    assert view.press(*a)
    view.move(*b)
    text, color = window.describe_gesture("throw", *[view.gesture[k] for k in ("start", "end")])
    assert text.startswith("0.20 m/s · limit 0.") and color == "good"
    view.release()
    attempt = window.session.attempt
    assert attempt["action"] == "throw"
    p = attempt["params"]
    assert p["distance_cm"] == pytest.approx(8.0, abs=0.1) and p["angle_deg"] == pytest.approx(180, abs=0.5)
    assert p["speed_m_s"] == pytest.approx(0.2, abs=0.005) and p["deviation_deg"] == pytest.approx(0, abs=0.5)
    assert window.view.tool == "throw"         # the tool stays active
    assert not view.press(5, 5)                # outside the plate: the view turns


def test_drop_and_push_with_the_mouse(window):
    window.tick()
    window.tool_buttons["drop"].click()
    window.s_height.setValue(20)
    assert window.l_height.text() == "Drop height: 20 cm"
    assert window.view.press(*window.view.to_screen((0.03, 0.02)))
    p = window.session.attempt["params"]
    assert window.session.attempt["action"] == "drop" and p["height_cm"] == pytest.approx(20)
    assert p["distance_cm"] == pytest.approx(100 * (0.03 ** 2 + 0.02 ** 2) ** 0.5, abs=0.1)
    window.b_center.click()
    play(window, 30)
    window.tool_buttons["push"].click()
    view = window.view
    ball = view.ball_on_plate()
    assert view.press(*view.to_screen(ball))
    view.move(*view.to_screen((ball[0], ball[1] + 0.02)))
    view.release()
    p = window.session.attempt["params"]
    assert window.session.attempt["action"] == "push"
    assert p["dvx_m_s"] == pytest.approx(0, abs=0.01) and p["dvy_m_s"] == pytest.approx(0.1, abs=0.01)
    assert not view.press(*view.to_screen((ball[0] + 0.1, ball[1])))     # far from the ball: the view turns


def test_setpoint(window):
    window.s_setpoint.buttons["circle"].click()
    assert window.session.setpoint == "circle" and len(window.view.overlay.path) > 100
    window.tick()
    play(window, 30)
    x, y = window.view.overlay.setpoint
    assert (x * x + y * y) ** 0.5 == pytest.approx(0.06, abs=1e-6)       # on the circle of 6 cm
    window.s_setpoint.buttons["center"].click()
    assert window.view.overlay.path == []


def test_center(window):
    window.tick()
    window.b_center.click()
    play(window, 5)
    assert window.session.attempt["action"] == "center"


def test_brain_changed(window):
    window.tick()
    window.throw(-0.10, 0.0, 0.4, 0.0)
    play(window, 5)
    window.s_brain.buttons["imitation"].click()
    assert isinstance(window.session.sim.brain, PolicyController)
    assert window.session.summary[-1]["brain"] == "pid" and window.session.attempt is None
    assert window.last_attempt_text().startswith("Throw · PID · interrupted after ")
    window.throw(-0.10, 0.0, 0.4, 0.0)
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
    assert window.state_text().startswith("The ball has fallen · back in ")
    assert window.view.overlay.state_color == COLORS["light"]["bad"]


def test_fall_followed_then_back_to_the_plate(window):
    window.tick()
    camera = window.view.camera
    home = (camera.lookat, camera.distance)
    window.throw(-0.10, 0.0, 1.2, 0.0)                    # too fast: falls beyond the edge, toward +x
    play(window, 60)                                       # 1 s
    assert window.session.state == "fallen"
    assert camera.lookat[0] > 0.05 and camera.distance > home[1]          # the camera follows the ball
    assert window.state_text().startswith("The ball has fallen · back in ")
    play(window, 90)                                       # 2.5 s after the throw: back on the plate
    assert window.session.state != "fallen" and window.session.summary[-1]["result"] == "fallen"
    assert window.session.summary[-1]["action"] == "throw"
    x, y = window.session.phys.ball()
    assert (x * x + y * y) ** 0.5 < 0.02
    play(window, 120)                                      # the camera is back where it was
    assert camera.lookat == home[0] and camera.distance == home[1]


def test_language(window):
    window.s_language.buttons["fr"].click()
    assert window.tool_buttons["throw"].text() == "Lancer"
    assert window.windowTitle() == "ball-plate-sim : robot équilibriste"
    assert window.readouts["speed_mm_s"].name.text() == "VITESSE"
    assert window.s_setpoint.buttons["square"].text() == "Carré"
    window.throw(-0.10, 0.0, 0.4, 0.0)
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
    window.throw(-0.10, 0.0, 0.4, 0.0)
    window.close()
    assert window.view.renderer is None and window.session.attempt is None
