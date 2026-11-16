"""The software (app/), without a screen: window, tools and gestures on the plate, setpoint, brain, measurements,
language, theme, pause, frames in step with real time, measurement of the cost of a frame (app/benchmark.py).

Skipped if PySide6 is not installed (pip install -r requirements-app.txt). Qt draws off screen; the 3D view is
rendered by MuJoCo, like the virtual camera.
"""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # before the application is created

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app import NAME
from app.i18n import Texts
from app.icon import SIZES as ICON_SIZES, ball_center, draw, icon
from app.settings import remembered, restore_geometry
from app.theme import COLORS
from app.view3d import RENDER_PIXELS
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
    assert window.l_height.text() == "Drop height: 20 cm" and window.view.drop_height == pytest.approx(0.2)
    view = window.view
    assert view.press(*view.to_screen((0.03, 0.02)))     # a click: dropped without speed
    view.release()
    p = window.session.attempt["params"]
    assert window.session.attempt["action"] == "drop" and p["height_cm"] == pytest.approx(20)
    assert p["distance_cm"] == pytest.approx(100 * (0.03 ** 2 + 0.02 ** 2) ** 0.5, abs=0.1)
    assert p["speed_m_s"] == 0 and p["deviation_deg"] == 0
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


def test_drop_with_a_speed(window):
    window.tick()
    window.tool_buttons["drop"].click()
    window.s_height.setValue(30)
    view = window.view
    assert view.press(*view.to_screen((-0.06, 0.0)))
    view.move(*view.to_screen((-0.02, 0.0)))                 # 4 cm: 0.2 m/s toward +x
    text, color = window.describe_gesture("drop", *[view.gesture[k] for k in ("start", "end")])
    assert text.startswith("0.20 m/s · arrives ") and text.endswith("° from the vertical") and color == "good"
    view.grab()                                              # drawn with its fall
    view.release()
    attempt = window.session.attempt
    p = attempt["params"]
    assert attempt["action"] == "drop" and p["height_cm"] == pytest.approx(30)
    assert p["speed_m_s"] == pytest.approx(0.2, abs=0.005) and p["angle_deg"] == pytest.approx(180, abs=0.5)
    assert p["deviation_deg"] == pytest.approx(0, abs=0.5)
    assert view.press(*view.to_screen((0.06, 0.0)))
    view.move(*view.to_screen((0.16, 0.0)))                  # 10 cm: 0.5 m/s from 6 cm, falling for 0.25 s
    text, color = window.describe_gesture("drop", *[view.gesture[k] for k in ("start", "end")])
    assert text == "0.50 m/s · lands off the plate" and color == "bad"
    view.release()


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


def test_settings(window):
    window.tick()
    start, end = (-0.10, 0.0), (0.0, 0.0)
    text, _ = window.describe_gesture("throw", start, end)
    assert text == "0.50 m/s · limit 0.71 m/s"               # the reference throw, ping-pong ball
    window.throw(-0.10, 0.0, 0.4, 0.0)
    play(window, 5)
    window.s_ball.buttons["solid"].click()                   # a setting changed: the attempt ends
    s = window.session
    assert s.ball == "solid" and s.attempt is None and s.summary[-1]["result"] == "interrupted"
    assert window.last_attempt_text().endswith("\nPing-pong · μ = 1.00 · 20 ms")
    text, _ = window.describe_gesture("throw", start, end)
    assert text == "0.50 m/s · limit 0.78 m/s"               # the solid ball rolls down faster: √(0.714/0.6) × 0.71
    window.s_friction.buttons["very_slippery"].click()
    assert s.cfg.mujoco_params.friction == 0.02
    window.s_latency.setValue(3)
    assert s.cfg.latency == 3 and window.l_latency.text() == "Latency of the camera: 60 ms"
    window.throw(-0.10, 0.0, 0.4, 0.0)
    window.s_latency.setValue(5)
    assert s.summary[-1]["latency_ms"] == 60 and s.cfg.latency == 5
    assert window.last_attempt_text().endswith("\nSolid · μ = 0.02 · 60 ms")
    window.tool_buttons["drop"].click()
    window.s_height.setValue(40)
    assert window.l_untrained.isVisibleTo(window)
    assert window.l_untrained.text() == "Not seen by the AI brain in training (5 to 35 cm)"
    window.s_height.setValue(35)
    assert not window.l_untrained.isVisibleTo(window)


def advance(window, duration: float) -> None:
    """Advances the sessions together, without rendering (faster than frames)."""
    for _ in range(round(duration / window.session.dt)):
        for r in window.robots:
            r.session.step()


def test_new_setpoint(window):
    window.tick()
    window.s_setpoint.buttons["circle"].click()                # not a disturbance: an attempt of its own
    assert window.state_text() == "New setpoint (PID) in progress"


def drag(view, points) -> None:
    """A gesture of the mouse through these points of the plate."""
    assert view.press(*view.to_screen(points[0]))
    for p in points[1:]:
        view.move(*view.to_screen(p))
    view.release()


def test_drawn_path(window):
    window.tick()
    window.s_setpoint.buttons["drawn"].click()                 # the tool Draw
    assert window.view.tool == "draw" and window.draw_row.isVisibleTo(window)
    assert window.session.setpoint == "center"                 # nothing drawn yet
    a = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)
    drag(window.view, [(0.05 * np.cos(t), 0.03 * np.sin(t)) for t in a])            # an ellipse
    s = window.session
    assert s.setpoint == "drawn" and s.attempt["action"] == "setpoint" and len(window.view.overlay.path) > 50
    assert window.l_draw.text().startswith("Path of 25 cm, one loop every 5.0 s.")    # ellipse 10 × 6 cm (25.5 cm
                                                               # round, a little less once smoothed), 5 cm/s
    key = s.path.key
    window.s_speed.setValue(8)                                 # faster: a new setpoint
    assert s.path.speed == 0.08 and s.path.key != key and s.summary[-1]["path"] == key
    drag(window.view, [(0.0, 0.0), (0.12, 0.0), (0.12, 0.02)])                        # too close to the edge
    assert s.path.speed == 0.08 and window.l_draw.text().startswith("Path not taken: it goes farther than 10 cm")
    drag(window.view, [(0.0, 0.0), (0.01, 0.0)])                                     # a click
    assert window.l_draw.text().startswith("Path too short")
    window.tool_buttons["throw"].click()                       # another tool: the path stays the setpoint
    assert window.view.tool == "throw" and s.setpoint == "drawn"
    window.b_split.click()                                     # side by side: the same path on both
    assert window.robots[1].session.path is s.path
    window.s_setpoint.buttons["center"].click()
    assert all(r.session.setpoint == "center" for r in window.robots) and window.view.overlay.path == []
    assert not window.draw_row.isVisibleTo(window)


def test_side_by_side(window):
    window.tick()
    window.s_ball.buttons["solid"].click()
    window.s_setpoint.buttons["circle"].click()
    window.b_split.click()
    pid, ai = window.robots
    assert window.split and (pid.brain, ai.brain) == ("pid", "imitation") and not window.s_brain.isEnabled()
    assert isinstance(ai.session.sim.brain, PolicyController) and ai.session.ball == "solid"
    assert ai.session.setpoint == "circle" and ai.session.t_setpoint == ai.session.t
    assert [r.session.attempt["action"] for r in window.robots] == ["center", "center"]       # they start together
    assert window.views.indexOf(pid.view) == 0 and window.views.indexOf(ai.view) == 1          # PID on the left
    window.s_setpoint.buttons["center"].click()
    window.throw(-0.10, 0.0, 0.4, 0.0)                         # every action on both
    assert [r.session.attempt["action"] for r in window.robots] == ["throw", "throw"]
    window.s_friction.buttons["slippery"].click()
    assert [r.session.cfg.mujoco_params.friction for r in window.robots] == [0.2, 0.2]
    pid.view.camera.turn(40, 0)
    pid.view.turned.emit(40.0, 0.0)                           # the views turn together
    assert ai.view.camera.azimuth == pid.view.camera.azimuth
    window.tick()
    play(window, 3)
    assert ai.view.pixels is not None and ai.view.overlay.state.startswith("AI · ")
    before = ai.view.pixels.copy()
    pid.view._create_renderer()                               # a view resized: the other one keeps its image
    ai.view.render_frame()
    assert np.array_equal(ai.view.pixels, before)
    window.throw(-0.10, 0.0, 0.4, 0.0)
    advance(window, 5.0)
    window.refresh()
    assert window.t_measures.isVisibleTo(window) and not window.w_readouts.isVisibleTo(window)
    assert window.t_measures.cells[0][1].text() != "—"         # distance to the setpoint, AI
    lines = window.last_attempt_text().split("\n")
    assert lines[0].startswith("Throw · PID · stabilized in ") and lines[1].startswith("Throw · AI · stabilized")
    assert window.t_compare.cells[0][0].text() == "1" and window.t_compare.cells[1][1].text() == "100 %"
    window.b_split.click()                                    # back to one robot: its comparison stays
    assert not window.split and window.s_brain.isEnabled() and window.robots == [pid]
    assert window.t_compare.cells[0][1].text() == "1"
    assert window.view.overlay.state == window.state_text()
    assert window.view.max_pixels == RENDER_PIXELS                 # side by side, each view had half of them


def test_views_share_the_pixels_rendered(window):
    """At most RENDER_PIXELS rendered, all views together: a view larger gets a smaller image, enlarged by Qt."""
    assert window.view.max_pixels == RENDER_PIXELS
    window.b_split.click()
    assert [r.view.max_pixels for r in window.robots] == [RENDER_PIXELS // 2] * 2
    window.view.resize(2000, 1400)
    window.view._create_renderer()
    assert window.view.renderer.width * window.view.renderer.height <= RENDER_PIXELS // 2


def test_choices_remembered(application, tmp_path):
    """The theme and the language once chosen, the size of the window at its closing: found at the next launch.
    Until chosen, those of the system; the setting of the robot is not kept."""
    kept = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    w = MainWindow(Texts("en"), mode="light", clock=FakeClock(), settings=kept)
    w.timer.stop()
    assert remembered(kept) == (None, None)                   # nothing chosen: the next launch follows the system
    w.s_language.buttons["fr"].click()
    w.s_mode.buttons["dark"].click()
    w.s_ball.buttons["heavy"].click()
    w.resize(760, 560)
    w.close()
    assert remembered(kept) == ("fr", "dark") and set(kept.allKeys()) == {"language", "theme", "geometry"}
    again = MainWindow(Texts("fr"), mode="dark", clock=FakeClock(), settings=kept)
    again.timer.stop()
    assert restore_geometry(again, kept) and (again.width(), again.height()) == (760, 560)
    again.close()
    kept.setValue("language", "de")                           # unknown to this version: that of the system
    assert remembered(kept) == (None, "dark")
    assert not restore_geometry(again, QSettings(str(tmp_path / "none.ini"), QSettings.IniFormat))


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


def test_panel_fits(window, application):
    """Nothing of the panel is cut or under its scroll bar, in either language (measured before: 318 pixels needed
    in English for 292 given, once the setpoint Drawing was added)."""
    for code in ("en", "fr"):
        window.s_language.buttons[code].click()
        for _ in range(3):
            application.processEvents()
        panel = window.scroll.widget()
        assert panel.minimumSizeHint().width() <= window.scroll.viewport().width(), code
        assert window.scroll.width() >= 300



def test_robot_shows_brain_ball_and_language(window):
    """On the robot: the LED of its brain lit, the printing of the ball in play, the label of the AI in the language of
    the interface; and the name on its front panel and in the title."""
    window.s_language.buttons["fr"].click()
    window.set_ball("heavy")
    window.set_brain("imitation")
    window.tick()
    m = window.view.scene.model
    shown = lambda name: m.geom_group[m.geom(name).id] == 0
    assert shown("deco_ball_print_heavy") and not shown("deco_ball_print_ping_pong")
    assert shown("deco_world_engraving_fr") and not shown("deco_world_engraving_en")
    assert m.mat_emission[m.material("part_led_ai").id] == 1.0 and m.mat_emission[m.material("part_led_pid").id] == 0.0
    window.set_mode("dark")                                # a new scene: the same parts shown at the next frame
    window.tick()
    m = window.view.scene.model
    assert m.geom_group[m.geom("deco_ball_print_heavy").id] == 0
    assert window.l_title.text() == NAME and window.windowTitle().startswith(NAME)

def test_language(window):
    window.s_language.buttons["fr"].click()
    assert window.tool_buttons["throw"].text() == "Lancer"
    assert window.windowTitle() == f"{NAME} : robot équilibriste"
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



def test_icon(application):
    """Drawn at every size of use: transparent in its corners (rounded square), the orange ball where it is."""
    assert [s.width() for s in icon().availableSizes()] == list(ICON_SIZES)
    x, y = ball_center()
    for size in (16, 64, 256):
        image = draw(size)
        ball = image.pixelColor(int((x + 0.04) * size), int((y + 0.04) * size))      # below the shine
        assert image.pixelColor(0, 0).alpha() == 0 and image.pixelColor(size // 2, size - 2).alpha() == 255
        assert ball.red() > 200 and 60 < ball.green() < 140 and ball.blue() < 60, (size, ball.getRgb())

def test_benchmark(application):
    from app.benchmark import QUALITIES, report, run
    sizes = ((64, 48),)
    results = run(sizes, frames=2)
    assert results["simulation"] > 0 and results["drawing"][sizes[0]] > 0
    assert set(results["rendering"]) == set(QUALITIES)
    assert all(row[sizes[0]] > 0 for row in results["rendering"].values())
    assert results["rendered"][sizes[0]] == sizes[0] and results["enlarged"][sizes[0]] > 0     # under the cap
    text = report(results, sizes)
    assert "64×48" in text and "look of D1" in text and "software, as it runs" in text


def test_close(window):
    window.tick()
    window.throw(-0.10, 0.0, 0.4, 0.0)
    window.close()
    assert window.view.renderer is None and window.session.attempt is None
