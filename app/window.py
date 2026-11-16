"""Main window of the software: the control panel on the left, the robot in 3D on the right.

A single loop, in the thread of the interface: a timer advances the simulation by the real time elapsed
(simulation.session.RealTime, up to MAX_LAG per frame), then renders the 3D view. The panel and the gestures of the
3D view act on the session directly, between two frames. The measurements are refreshed every READOUT_EVERY frames
(readable, and cheaper).

The tools of the ball (throw, drop, push) stay active until clicked again; with one active, the mouse acts on the
plate (app/view3d.py). The physical limit shown while throwing is that of the ball in play (c = 1/(1+k)). A drop with a speed shows where the ball lands and its angle
of arrival, in the good color if it lands on the plate, in the bad color if not; no limit is shown for it (not
computed: bounce, then rolling): on the plate does not mean caught.

When the ball falls, the camera glides to look between the robot and the ball, and moves back to keep both in view
(the ball is followed up to FOLLOW_REACH from the center of the plate), for FALL_SHOW of simulated time; then the
ball comes back to the center of the plate, and the camera to where it was. A pause stops all of it.
"""
import math
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QMainWindow, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from app.gestures import MIN_SPEED, drop_arguments, drop_flight, throw_arguments, throw_limit, velocity
from app.i18n import LANGUAGES, Texts
from app.readouts import distance_history, readouts
from app.scene import DisplayScene
from app.theme import COLORS, MODES, load_fonts, scene_style, stylesheet, system_mode
from app.view3d import View3D
from app.widgets import Card, Readout, Segmented, Sparkline, label
from controller.policy import MAX_LATENCY
from simulation.config import G
from simulation.session import BALLS, FRICTIONS, SETPOINTS, RealTime, Session, sandbox_config
from training.environment import DROP_HEIGHT as TRAINED_HEIGHT

FRAMES_PER_SECOND = 60
MAX_LAG = 0.1           # (s) of real time caught up per frame at most: the simulation stays in step with real time as
                        # long as a frame takes less (measured: up to 45 ms on an integrated graphics card)
READOUT_EVERY = 5                                           # frames
BRAINS = ("pid", "imitation")                               # the PID brain, the AI brain (brains/imitation.npz)
TOOLS = ("throw", "drop", "push")
DROP_HEIGHT = (0, 40, 30)                                   # (cm) range and start value of the drop height
                                                            # (the AI brain saw TRAINED_HEIGHT: shown when outside)
FRICTION_NAMES = dict(zip(FRICTIONS, ("normal", "slippery", "very_slippery")))
# The settings stay within what the AI brain saw in training: balls and frictions (simulation/session.py), latency
# from 1 to MAX_LATENCY control periods (20 to 100 ms), which is also the most it can take.
PATH_DURATION = 8.0                                         # (s) of setpoint drawn: a square, two circles
FALL_SHOW = 2.0                                             # (s) of simulated time showing the fallen ball
FOLLOW_REACH = 2.5                                          # (m) the camera follows the ball this far at most
                                                            # (a fast ball rolls about 2 m on the floor in 2 s)
FOLLOW_TIME = 0.3                                           # (s) time constant of the moves of the camera
PANEL_WIDTH = 300                                           # (pixels)
# The measurements of the panel: (key of readouts(), key of the text, unit, decimals)
MEASURES = (("distance_mm", "distance", "mm", 1), ("speed_mm_s", "speed", "mm/s", 0),
            ("tilt_x_deg", "tilt_x", "°", 2), ("tilt_y_deg", "tilt_y", "°", 2),
            ("height_mm", "height", "mm", 1), ("torque_Nm", "torque", "N·m", 3))


class MainWindow(QMainWindow):
    def __init__(self, texts: Texts, brain: str = "pid", mode: str = None, clock=time.perf_counter):
        """mode: "light" or "dark" (None: that of the system). clock: real time (s), at every frame (replaced by a
        simulated clock in the tests)."""
        super().__init__()
        load_fonts()
        self.texts = texts
        self.mode = mode or system_mode()
        self.clock = clock
        self.session = Session(replace(sandbox_config(), brain=brain), record=False)
        self.real_time = RealTime(MAX_LAG)
        cfg = self.session.cfg
        self.view = View3D(DisplayScene(cfg, scene_style(self.mode)), self.session.phys.data,
                           cfg.geometry.height - 0.035, cfg.plate.plate_radius, cfg.plate.ball_radius)
        self.view.describe = self.describe_gesture
        self.view.thrown.connect(self.throw)
        self.view.dropped.connect(self.drop)
        self.view.pushed.connect(self.push)
        self._scene_mode = self.mode                        # theme of the display scene
        self.tool = None
        self.fall_time = None                               # simulated time of the fall, while the ball is down
        self._home = None                                   # (lookat, distance) of the camera before the fall
        self._last_t = self.session.t
        self.frames = 0
        self._build()
        self.set_setpoint(self.session.setpoint)
        self.set_mode(self.mode)
        self.retranslate()
        self.timer = QTimer(self, timerType=Qt.PreciseTimer, interval=round(1000 / FRAMES_PER_SECOND))
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # --- Layout ------------------------------------------------------------------------------
    def _build(self) -> None:
        self.l_title = label("title")
        self.l_subtitle = label("subtitle")

        self.c_brain = Card()
        self.s_brain = Segmented(BRAINS, self.session.cfg.brain)
        self.s_brain.changed.connect(self.set_brain)
        self.l_brain = label("caption", wordWrap=True)
        self.c_brain.body.addWidget(self.s_brain)
        self.c_brain.body.addWidget(self.l_brain)

        self.c_ball = Card()
        self.tool_buttons = {}
        tools = QHBoxLayout()
        tools.setSpacing(6)
        for name in TOOLS:
            b = QPushButton(checkable=True, objectName="tool")
            b.clicked.connect(lambda _=False, name=name: self.set_tool(name))
            tools.addWidget(b)
            self.tool_buttons[name] = b
        self.c_ball.body.addLayout(tools)
        self.height_row = QWidget()
        height = QVBoxLayout(self.height_row)
        height.setContentsMargins(0, 2, 0, 2)
        height.setSpacing(4)
        self.l_height = label("caption")
        self.s_height = QSlider(Qt.Horizontal, minimum=DROP_HEIGHT[0], maximum=DROP_HEIGHT[1], value=DROP_HEIGHT[2])
        self.s_height.valueChanged.connect(lambda _: self._show_height())
        self.l_untrained = label("caption", wordWrap=True)        # outside the heights of the training
        height.addWidget(self.l_height)
        height.addWidget(self.s_height)
        height.addWidget(self.l_untrained)
        self.height_row.setVisible(False)
        self.c_ball.body.addWidget(self.height_row)
        self.b_center = QPushButton(clicked=self.center)
        self.b_pause = QPushButton(checkable=True, toggled=self.set_pause, objectName="ghost")
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self.b_center)
        actions.addWidget(self.b_pause)
        self.c_ball.body.addLayout(actions)

        self.c_setpoint = Card()
        self.s_setpoint = Segmented(tuple(SETPOINTS), self.session.setpoint)
        self.s_setpoint.changed.connect(self.set_setpoint)
        self.l_setpoint = label("caption", wordWrap=True)
        self.c_setpoint.body.addWidget(self.s_setpoint)
        self.c_setpoint.body.addWidget(self.l_setpoint)

        self.c_settings = Card()
        cfg = self.session.cfg
        self.l_ball = label("caption")
        self.s_ball = Segmented(tuple(BALLS), self.session.ball)
        self.s_ball.changed.connect(self.set_ball)
        self.l_friction = label("caption")
        self.s_friction = Segmented(tuple(FRICTION_NAMES.values()), FRICTION_NAMES.get(cfg.mujoco_params.friction))
        self.s_friction.changed.connect(self.set_friction)
        self.l_latency = label("caption")
        self.s_latency = QSlider(Qt.Horizontal, minimum=1, maximum=MAX_LATENCY, value=cfg.latency, pageStep=1)
        self.s_latency.valueChanged.connect(self.set_latency)
        self.l_settings = label("caption", wordWrap=True)
        for w in (self.l_ball, self.s_ball, self.l_friction, self.s_friction, self.l_latency, self.s_latency,
                  self.l_settings):
            self.c_settings.body.addWidget(w)

        self.c_measures = Card()
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.readouts = {}
        for i, (key, *_) in enumerate(MEASURES):
            self.readouts[key] = Readout()
            grid.addWidget(self.readouts[key], i // 2, i % 2)
        self.c_measures.body.addLayout(grid)
        self.sparkline = Sparkline()
        self.l_history = label("caption")
        self.c_measures.body.addWidget(self.sparkline)
        self.c_measures.body.addWidget(self.l_history)

        self.c_last = Card(spacing=6)
        self.l_last = label("body", wordWrap=True)
        self.c_last.body.addWidget(self.l_last)

        self.s_mode = Segmented(MODES, self.mode)
        self.s_mode.changed.connect(self.set_mode)
        self.s_language = Segmented(LANGUAGES, self.texts.code)
        self.s_language.set_texts({code: code.upper() for code in LANGUAGES})
        self.s_language.changed.connect(self.set_language)

        panel = QWidget(objectName="window")
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 4, 0)
        column.setSpacing(10)
        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(self.l_title)
        header.addWidget(self.l_subtitle)
        column.addLayout(header)
        for w in (self.c_brain, self.c_ball, self.c_setpoint, self.c_settings, self.c_measures, self.c_last):
            column.addWidget(w)
        column.addStretch(1)
        settings = QHBoxLayout()
        settings.addWidget(self.s_mode, 3)
        settings.addWidget(self.s_language, 2)
        column.addLayout(settings)
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame,
                             horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setFixedWidth(PANEL_WIDTH)

        central = QWidget(objectName="window")
        row = QHBoxLayout(central)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(12)
        row.addWidget(scroll)
        row.addWidget(self.view, 1)
        self.setCentralWidget(central)

    def retranslate(self) -> None:
        """Writes all the texts in the current language."""
        tr = self.texts
        self.setWindowTitle(tr("window.title"))
        self.l_title.setText(tr("app.name"))
        self.l_subtitle.setText(tr("app.subtitle"))
        self.c_brain.set_title(tr("section.brain"))
        self.s_brain.set_texts({name: tr(f"brain.{name}") for name in BRAINS})
        self.l_brain.setText(tr("caption.brain"))
        self.c_ball.set_title(tr("section.ball"))
        for name, b in self.tool_buttons.items():
            b.setText(tr(f"button.{name}"))
            b.setToolTip(tr(f"tip.{name}"))
        self.b_center.setText(tr("button.center"))
        self.b_center.setToolTip(tr("tip.center"))
        self.b_pause.setText(tr("button.resume" if self.real_time.paused else "button.pause"))
        self._show_height()
        self.c_setpoint.set_title(tr("section.setpoint"))
        self.s_setpoint.set_texts({name: tr(f"setpoint.{name}") for name in SETPOINTS})
        self.l_setpoint.setText(tr("caption.setpoint"))
        self.c_settings.set_title(tr("section.settings"))
        self.l_ball.setText(tr("label.ball"))
        self.s_ball.set_texts({name: tr(f"ball.{name}") for name in BALLS})
        for name, b in self.s_ball.buttons.items():
            b.setToolTip(tr(f"tip.ball.{name}"))
        self.l_friction.setText(tr("label.friction"))
        self.s_friction.set_texts({name: tr(f"friction.{name}") for name in FRICTION_NAMES.values()})
        for mu, name in FRICTION_NAMES.items():
            self.s_friction.buttons[name].setToolTip(tr("tip.friction", mu=tr.number(mu, 2)))
        self.s_latency.setToolTip(tr("tip.latency"))
        self._show_latency()
        self.l_settings.setText(tr("caption.settings"))
        self.c_measures.set_title(tr("section.measures"))
        self.c_measures.setToolTip(tr("tip.measures"))
        for key, name, unit, _ in MEASURES:
            self.readouts[key].set_name(tr(f"measure.{name}").upper(), "" if unit == "°" else unit)  # 12,3°
        self.l_history.setText(tr("caption.history"))
        self.c_last.set_title(tr("section.last"))
        self.s_mode.set_texts({m: tr(f"theme.{m}") for m in MODES})
        self.view.overlay.hint = tr(f"hint.{self.tool or 'view'}")
        self.refresh()

    def _show_height(self) -> None:
        height = self.s_height.value()
        self.l_height.setText(self.texts("label.height", height=height))
        low, high = (round(100 * h) for h in TRAINED_HEIGHT)
        self.l_untrained.setText(self.texts("caption.untrained_height", low=low, high=high))
        self.l_untrained.setVisible(not low <= height <= high)
        self.view.drop_height = self.drop_height()

    def _show_latency(self) -> None:
        self.l_latency.setText(self.texts("label.latency", latency=round(1000 * self.session.cfg.latency
                                                                         * self.session.cfg.control_period)))

    # --- Actions -----------------------------------------------------------------------------
    def set_tool(self, name) -> None:
        """Activates the tool `name`, or deactivates it if it is the active one."""
        self.tool = None if name == self.tool else name
        for n, b in self.tool_buttons.items():
            b.setChecked(n == self.tool)
        self.height_row.setVisible(self.tool == "drop")
        self.view.set_tool(self.tool)
        self.view.overlay.hint = self.texts(f"hint.{self.tool or 'view'}")

    def throw(self, x: float, y: float, vx: float, vy: float) -> None:
        self.session.throw(**throw_arguments((x, y), vx, vy))

    def drop_height(self) -> float:
        """(m) Height of a drop: of the bottom of the ball above the plate."""
        return self.s_height.value() / 100

    def drop(self, x: float, y: float, vx: float, vy: float) -> None:
        self.session.throw(**drop_arguments((x, y), self.drop_height(), vx, vy))

    def push(self, dvx: float, dvy: float) -> None:
        self.session.push(dvx, dvy)

    def center(self) -> None:
        self.session.center_ball()

    def set_brain(self, name: str) -> None:
        self.session.set_brain(name)

    def set_ball(self, name: str) -> None:
        self.session.set_ball(name)

    def set_friction(self, name: str) -> None:
        self.session.set_friction(next(mu for mu, n in FRICTION_NAMES.items() if n == name))

    def set_latency(self, periods: int) -> None:
        self.session.set_latency(periods)
        self._show_latency()

    def set_setpoint(self, name: str) -> None:
        """Center, circle or square, drawn on the plate."""
        self.session.set_setpoint(name)
        f = SETPOINTS[name]
        path = [f(t)[:2] for t in np.linspace(0.0, PATH_DURATION, 161)]
        self.view.overlay.path = path if np.ptp(np.array(path), axis=0).max() > 0 else []

    def set_pause(self, paused: bool) -> None:
        self.real_time.paused = paused
        self.b_pause.setText(self.texts("button.resume" if paused else "button.pause"))
        self._update_overlay()
        self.view.update()

    def set_language(self, code: str) -> None:
        self.texts.set_language(code)
        self.retranslate()

    def set_mode(self, mode: str) -> None:
        """Light or dark theme: the widgets, and the 3D scene, whose floor continues the background."""
        self.mode = mode
        self.setStyleSheet(stylesheet(mode))
        if mode != self._scene_mode:
            self.view.set_scene(DisplayScene(self.session.cfg, scene_style(mode)))
            self._scene_mode = mode
        self.view.overlay.colors = COLORS[mode]
        self.sparkline.set_values(self.sparkline.values, COLORS[mode]["accent"])
        self.view.update()

    def describe_gesture(self, kind: str, start, end) -> tuple:
        """Text and color (key of the theme) of the arrow of a gesture: speed, and for a throw its physical limit
        (good color within it, bad color beyond); for a drop with a speed, the angle of arrival from the vertical
        (good color if the ball lands on the plate, bad color if not)."""
        tr, cfg = self.texts, self.session.cfg
        vx, vy = velocity(start, end)
        speed = math.hypot(vx, vy)
        if kind == "push":
            return tr("gesture.push", speed=tr.number(speed, 2)), "accent"
        if kind == "drop":
            if speed < MIN_SPEED:
                return "", "accent"
            landing, angle, _ = drop_flight(start, vx, vy, self.drop_height(),
                                            self.session.phys.data.body("plate").xmat)
            if math.hypot(*landing) > cfg.plate.plate_radius:
                return tr("gesture.drop_off", speed=tr.number(speed, 2)), "bad"
            return tr("gesture.drop", speed=tr.number(speed, 2), angle=tr.number(angle, 0)), "good"
        if speed == 0:
            return "", "accent"
        limit = throw_limit(start, vx, vy, cfg.plate.plate_radius, cfg.theta_max, self.session.ball_coefficient * G)
        return (tr("gesture.throw", speed=tr.number(speed, 2), limit=tr.number(limit, 2)),
                "good" if speed <= limit else "bad")

    # --- Loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        """One frame: the simulation catches up with real time, then the 3D view (and, from time to time, the
        measurements) are updated."""
        self.real_time.advance(self.session, self.clock())
        self._follow_fall()
        self.frames += 1
        if self.frames % READOUT_EVERY == 0:
            self.refresh()
        else:
            self._update_overlay()
        self.view.render_frame()

    def _follow_fall(self) -> None:
        """While the ball is down: the camera follows it, then the ball comes back to the center after FALL_SHOW.
        Afterwards, the camera goes back to where it was."""
        s, camera = self.session, self.view.camera
        dt, self._last_t = s.t - self._last_t, s.t
        if s.state == "fallen":
            if self.fall_time is None:
                self.fall_time, self._home = s.t, (camera.lookat, camera.distance)
            home_lookat, home_distance = self._home
            ball, plate = s.phys.data.body("ball").xpos, s.phys.data.body("plate").xpos
            dx, dy = ball[0] - plate[0], ball[1] - plate[1]
            reach = math.hypot(dx, dy)
            f = min(1.0, FOLLOW_REACH / reach) if reach > 0 else 0.0
            camera.glide((plate[0] + f * dx / 2, plate[1] + f * dy / 2, home_lookat[2]), home_distance + f * reach,
                         dt, FOLLOW_TIME)                   # halfway between the robot and the ball
            if s.t - self.fall_time >= FALL_SHOW:
                s.center_ball()
            return
        self.fall_time = None
        if self._home is not None:                          # back to the view of before the fall
            home_lookat, home_distance = self._home
            camera.glide(home_lookat, home_distance, dt, FOLLOW_TIME)
            if max(abs(a - b) for a, b in zip(camera.lookat, home_lookat)) < 1e-3:    # 1 mm: there
                camera.lookat, camera.distance, self._home = home_lookat, home_distance, None

    def refresh(self) -> None:
        """Measurements, curve, last attempt and texts over the 3D view."""
        tr = self.texts
        values = readouts(self.session)
        for key, _, unit, decimals in MEASURES:
            v = values[key]
            text = "—" if v is None else tr.number(v, decimals)
            self.readouts[key].value.setText(text + "°" if unit == "°" and v is not None else text)
        self.sparkline.set_values(distance_history(self.session))
        self.l_last.setText(self.last_attempt_text())
        self._update_overlay(values)

    def _update_overlay(self, values: dict = None) -> None:
        tr, s, rt, c = self.texts, self.session, self.real_time, COLORS[self.mode]
        o = self.view.overlay
        o.state = tr("status.paused") if rt.paused else self.state_text()
        o.state_color = c["text3"] if rt.paused else c["bad"] if s.state == "fallen" else c["good"]
        o.bottom_right = tr("overlay.clock", time=tr.number(s.t, 1), speed=tr.number(rt.speed, 2),
                            fps=tr.number(rt.frames_per_second, 0))
        o.setpoint = tuple(s.sim.cfg.trajectory(s.t)[:2])
        if values is not None:
            o.bottom_left = ("" if values["x_mm"] is None else
                             tr("overlay.position", x=tr.number(values["x_mm"], 1), y=tr.number(values["y_mm"], 1)))

    def state_text(self) -> str:
        """What the robot is doing."""
        tr, s = self.texts, self.session
        if s.state == "fallen":
            left = FALL_SHOW - (s.t - self.fall_time) if self.fall_time is not None else FALL_SHOW
            return tr("state.fallen", time=tr.number(max(left, 0.0), 1))
        if s.attempt is not None:
            a = s.attempt
            return tr("state.attempt", action=tr(f"action.{a['action']}"), brain=tr(f"brain.{a['brain']}"))
        return tr("state.ready")

    def last_attempt_text(self) -> str:
        """Result of the last finished attempt."""
        tr = self.texts
        if not self.session.summary:
            return tr("last.none")
        row = self.session.summary[-1]
        duration = row["stabilized_in_s"] if row["result"] == "stabilized" else row["duration_s"]
        result = tr("result." + row["result"].replace(" ", "_"), time=tr.number(duration, 2))
        settings = tr("last.settings", ball=tr(f"ball.{row['ball']}"), mu=tr.number(row["friction"], 2),
                      latency=row["latency_ms"])
        return (tr("last.row", action=tr(f"action.{row['action']}"), brain=tr(f"brain.{row['brain']}"), result=result)
                + "\n" + settings)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.view.close_renderer()
        self.session.close()
        super().closeEvent(event)
