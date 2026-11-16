"""Main window of the software: the control panel on the left, the robot in 3D on the right; or two robots side by
side, one per brain, to compare them.

A single loop, in the thread of the interface: a timer advances the simulations by the real time elapsed
(simulation.session.RealTime, up to MAX_LAG per frame), then renders the 3D views. The panel and the gestures act on
the sessions directly, between two frames. The measurements are refreshed every READOUT_EVERY frames (readable, and
cheaper).

The tools of the ball (throw, drop, push) stay active until clicked again; with one active, the mouse acts on the
plate (app/view3d.py). The physical limit shown while throwing is that of the ball in play (c = 1/(1+k)). A drop
with a speed shows where the ball lands and its angle of arrival, in the good color if it lands on the plate, in the
bad color if not; no limit is shown for it (not computed: bounce, then rolling): on the plate does not mean caught.

Side by side: the PID brain on the left, the AI brain on the right, each robot with its own session. Every action (a
gesture on either view, the setpoint, a setting, Center) is applied to both, the views turn and zoom together, and the
sessions advance step by step together. The panel shows the measurements of both. The comparison of the brains
(app/comparison.py) covers the attempts of the whole session, side by side or not.

When a ball falls, the camera of its view follows it, then the ball comes back to the center (app/robot.py).

Setpoint "drawn": the tool Draw traces a path on the plate, which becomes the setpoint (simulation.config.DrawnPath),
traveled in a loop at the speed chosen. A path that goes farther than DRAW_RADIUS from the center is refused.
"""
import math
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QMainWindow, QPushButton, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from app.comparison import compare
from app.gestures import MIN_SPEED, drop_arguments, drop_flight, throw_arguments, throw_limit, velocity
from app.i18n import LANGUAGES, Texts
from app.readouts import distance_history, readouts
from app.robot import Robot
from app.scene import DisplayScene
from app.theme import COLORS, MODES, load_fonts, scene_style, stylesheet, system_mode
from app.view3d import View3D
from app.widgets import Card, Readout, Segmented, Sparkline, Table, Wrapped, label
from controller.policy import MAX_LATENCY
from simulation.config import G, DrawnPath
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
SETPOINT_NAMES = (*SETPOINTS, "drawn")
DRAW_RADIUS = 0.10      # (m) a drawn path must stay within this distance of the center: the ball (2 cm) then keeps
                        # 3 cm from the edge of the plate (15 cm)
MIN_DRAW_LENGTH = 0.03  # (m) a shorter path is a click
DRAW_SPEED = (2, 10, 5)                                     # (cm/s) range and start value of the speed on a path
PANEL_WIDTH = 300       # (pixels) at least: wider if its widest row needs it (texts and fonts change with the
                        # language and the system; measured: 318 needed in English with the setpoint Drawing)
# The measurements of the panel: (key of readouts(), key of the text, unit, decimals)
MEASURES = (("distance_mm", "distance", "mm", 1), ("speed_mm_s", "speed", "mm/s", 0),
            ("tilt_x_deg", "tilt_x", "°", 2), ("tilt_y_deg", "tilt_y", "°", 2),
            ("height_mm", "height", "mm", 1), ("torque_Nm", "torque", "N·m", 3))
# The comparison of the brains: (key of app.comparison.compare, key of the text, unit, decimals)
INDICATORS = (("attempts", "attempts", "", 0), ("caught_pct", "caught", "%", 0),
              ("stabilized_s", "stabilized", "s", 2), ("margin_mm", "margin", "mm", 0))


class _Together:
    """The sessions of the robots, advanced step by step together (for RealTime: the time is that of the first)."""

    def __init__(self, sessions):
        self.sessions = sessions

    @property
    def t(self) -> float:
        return self.sessions[0].t

    def step(self) -> None:
        for s in self.sessions:
            s.step()


class MainWindow(QMainWindow):
    def __init__(self, texts: Texts, brain: str = "pid", mode: str = None, clock=time.perf_counter):
        """mode: "light" or "dark" (None: that of the system). clock: real time (s), at every frame (replaced by a
        simulated clock in the tests)."""
        super().__init__()
        load_fonts()
        self.texts = texts
        self.mode = mode or system_mode()
        self.clock = clock
        self.real_time = RealTime(MAX_LAG)
        self._scene_mode = self.mode                        # theme of the display scenes
        self.tool = None
        self.frames = 0
        self.robots = []                                    # the first one is always there: its time is the window's
        self.past_rows = []                                 # attempts of the robots taken away (side by side ended)
        self.path = []                                      # path of the setpoint, drawn on the plates
        self.drawn_path = None                              # the last path drawn (DrawnPath)
        self.views = QHBoxLayout()
        self.views.setSpacing(12)
        self._add_robot(Session(replace(sandbox_config(), brain=brain), record=False))
        self._build()
        self.set_setpoint(self.session.setpoint)
        self.set_mode(self.mode)
        self.retranslate()
        self.timer = QTimer(self, timerType=Qt.PreciseTimer, interval=round(1000 / FRAMES_PER_SECOND))
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    @property
    def session(self) -> Session:
        """Session of the first robot (the only one, unless side by side)."""
        return self.robots[0].session

    @property
    def view(self) -> View3D:
        return self.robots[0].view

    @property
    def split(self) -> bool:
        """Two robots side by side."""
        return len(self.robots) > 1

    def _ordered(self) -> list:
        """The robots in the order of BRAINS (side by side: PID on the left)."""
        return sorted(self.robots, key=lambda r: BRAINS.index(r.brain))

    # --- Robots ------------------------------------------------------------------------------
    def _add_robot(self, session: Session) -> Robot:
        cfg = session.cfg
        view = View3D(DisplayScene(cfg, scene_style(self._scene_mode)), session.phys.data,
                      cfg.geometry.height - 0.035, cfg.plate.plate_radius, cfg.plate.ball_radius)
        robot = Robot(session, view)
        view.describe = lambda kind, start, end: self.describe_gesture(kind, start, end, robot)
        view.thrown.connect(self.throw)
        view.dropped.connect(self.drop)
        view.pushed.connect(self.push)
        view.drawn.connect(self.draw)
        view.draw_radius = DRAW_RADIUS
        view.turned.connect(lambda dx, dy: [r.view.camera.turn(dx, dy) for r in self.robots if r.view is not view])
        view.zoomed.connect(lambda notches: [r.view.camera.zoom(notches) for r in self.robots if r.view is not view])
        if self.robots:                                     # the same point of view as the first one, and its look
            camera, first = view.camera, self.robots[0]
            camera.azimuth, camera.elevation = first.view.camera.azimuth, first.view.camera.elevation
            camera.lookat, camera.distance = first.home()
            view.set_tool(self.tool)
            view.drop_height = self.drop_height()
            view.overlay.colors = COLORS[self.mode]
            view.overlay.path = self.path
            view.overlay.hint = self.texts(f"hint.{self.tool or 'view'}")
        self.robots.append(robot)
        self._place_views()
        return robot

    def _place_views(self) -> None:
        for r in self.robots:
            self.views.removeWidget(r.view)
        for r in self._ordered():
            self.views.addWidget(r.view, 1)

    def set_split(self, on: bool) -> None:
        """Both brains side by side, or only the first robot. A second robot starts with the settings and the
        setpoint of the first; both balls are then put at the center, the setpoints from their beginning: they
        start together."""
        if on == self.split:
            return
        if on:
            first = self.session
            other = next(b for b in BRAINS if b != first.cfg.brain)
            self._add_robot(Session(replace(sandbox_config(), brain=other, latency=first.cfg.latency,
                                            latency_compensation=first.cfg.latency_compensation,
                                            mujoco_params=first.cfg.mujoco_params), record=False))
            for r in self.robots:                           # the setpoints from their beginning, together
                if first.setpoint == "drawn":
                    r.session.set_path(first.path)
                else:
                    r.session.set_setpoint(first.setpoint)
            self.center()
        else:
            robot = self.robots.pop()
            self.past_rows += robot.session.close()
            robot.view.close_renderer()
            self.views.removeWidget(robot.view)
            robot.view.deleteLater()
        self.b_split.setChecked(on)
        self.s_brain.setEnabled(not on)
        self.retranslate()

    # --- Layout ------------------------------------------------------------------------------
    def _build(self) -> None:
        cfg = self.session.cfg
        self.l_title = label("title")
        self.l_subtitle = label("subtitle")

        self.c_brain = Card()
        self.s_brain = Segmented(BRAINS, cfg.brain)
        self.s_brain.changed.connect(self.set_brain)
        self.b_split = QPushButton(checkable=True, objectName="ghost")
        self.b_split.toggled.connect(self.set_split)
        self.l_brain = Wrapped("caption")
        self.c_brain.body.addWidget(self.s_brain)
        self.c_brain.body.addWidget(self.b_split)
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
        self.l_untrained = Wrapped("caption")        # outside the heights of the training
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
        self.s_setpoint = Segmented(SETPOINT_NAMES, self.session.setpoint)
        self.s_setpoint.changed.connect(self.set_setpoint)
        self.l_setpoint = Wrapped("caption")
        self.draw_row = QWidget()                           # shown with the setpoint "drawn"
        draw = QVBoxLayout(self.draw_row)
        draw.setContentsMargins(0, 2, 0, 2)
        draw.setSpacing(4)
        self.l_speed = label("caption")
        self.s_speed = QSlider(Qt.Horizontal, minimum=DRAW_SPEED[0], maximum=DRAW_SPEED[1], value=DRAW_SPEED[2],
                               pageStep=1)
        self.s_speed.valueChanged.connect(self.set_draw_speed)
        self.l_draw = Wrapped("caption")
        for w in (self.l_speed, self.s_speed, self.l_draw):
            draw.addWidget(w)
        self.draw_row.setVisible(False)
        self.c_setpoint.body.addWidget(self.s_setpoint)
        self.c_setpoint.body.addWidget(self.draw_row)
        self.c_setpoint.body.addWidget(self.l_setpoint)

        self.c_settings = Card()
        self.l_ball = label("caption")
        self.s_ball = Segmented(tuple(BALLS), self.session.ball)
        self.s_ball.changed.connect(self.set_ball)
        self.l_friction = label("caption")
        self.s_friction = Segmented(tuple(FRICTION_NAMES.values()), FRICTION_NAMES.get(cfg.mujoco_params.friction))
        self.s_friction.changed.connect(self.set_friction)
        self.l_latency = label("caption")
        self.s_latency = QSlider(Qt.Horizontal, minimum=1, maximum=MAX_LATENCY, value=cfg.latency, pageStep=1)
        self.s_latency.valueChanged.connect(self.set_latency)
        self.l_settings = Wrapped("caption")
        for w in (self.l_ball, self.s_ball, self.l_friction, self.s_friction, self.l_latency, self.s_latency,
                  self.l_settings):
            self.c_settings.body.addWidget(w)

        self.c_measures = Card()
        self.w_readouts = QWidget()                         # one robot: the values in large
        grid = QGridLayout(self.w_readouts)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.readouts = {}
        for i, (key, *_) in enumerate(MEASURES):
            self.readouts[key] = Readout()
            grid.addWidget(self.readouts[key], i // 2, i % 2)
        self.t_measures = Table(len(MEASURES), len(BRAINS))   # side by side: a column per brain
        self.c_measures.body.addWidget(self.w_readouts)
        self.c_measures.body.addWidget(self.t_measures)
        self.sparkline = Sparkline()
        self.l_history = label("caption")
        self.c_measures.body.addWidget(self.sparkline)
        self.c_measures.body.addWidget(self.l_history)

        self.c_last = Card(spacing=6)
        self.l_last = Wrapped("body")
        self.c_last.body.addWidget(self.l_last)

        self.c_compare = Card()
        self.t_compare = Table(len(INDICATORS), len(BRAINS))
        self.l_compare = Wrapped("caption")
        self.c_compare.body.addWidget(self.t_compare)
        self.c_compare.body.addWidget(self.l_compare)

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
        for w in (self.c_brain, self.c_ball, self.c_setpoint, self.c_settings, self.c_measures, self.c_last,
                  self.c_compare):
            column.addWidget(w)
        column.addStretch(1)
        settings = QHBoxLayout()
        settings.addWidget(self.s_mode, 3)
        settings.addWidget(self.s_language, 2)
        column.addLayout(settings)
        self.scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame,
                                  horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(panel)
        self.scroll.setFixedWidth(PANEL_WIDTH)
        panel.installEventFilter(self)                      # its layout changes: _fit_panel

        central = QWidget(objectName="window")
        row = QHBoxLayout(central)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(12)
        row.addWidget(self.scroll)
        row.addLayout(self.views, 1)
        self.setCentralWidget(central)

    def retranslate(self) -> None:
        """Writes all the texts in the current language (and shows what depends on side by side)."""
        tr, c = self.texts, COLORS[self.mode]
        self.setWindowTitle(tr("window.title"))
        self.l_title.setText(tr("app.name"))
        self.l_subtitle.setText(tr("app.subtitle"))
        self.c_brain.set_title(tr("section.brain"))
        self.s_brain.set_texts({name: tr(f"brain.{name}") for name in BRAINS})
        self.b_split.setText(tr("button.split"))
        self.b_split.setToolTip(tr("tip.split"))
        self.l_brain.setText(tr("caption.split" if self.split else "caption.brain"))
        self.c_ball.set_title(tr("section.ball"))
        for name, b in self.tool_buttons.items():
            b.setText(tr(f"button.{name}"))
            b.setToolTip(tr(f"tip.{name}"))
        self.b_center.setText(tr("button.center"))
        self.b_center.setToolTip(tr("tip.center"))
        self.b_pause.setText(tr("button.resume" if self.real_time.paused else "button.pause"))
        self._show_height()
        self.c_setpoint.set_title(tr("section.setpoint"))
        self.s_setpoint.set_texts({name: tr(f"setpoint.{name}") for name in SETPOINT_NAMES})
        self.s_setpoint.buttons["drawn"].setToolTip(tr("tip.drawn"))
        self.l_speed.setText(tr("label.speed", speed=self.s_speed.value()))
        self._show_drawing()
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
        self.t_measures.set_names([tr(f"measure.{name}") + ("" if unit == "°" else f" ({unit})")
                                   for _, name, unit, _ in MEASURES])
        self.w_readouts.setVisible(not self.split)
        self.t_measures.setVisible(self.split)
        self.l_history.setText(tr("caption.history"))
        self.c_last.set_title(tr("section.last"))
        self.c_compare.set_title(tr("section.compare"))
        self.t_compare.set_names([tr(f"compare.{name}") for _, name, _, _ in INDICATORS])
        for table in (self.t_measures, self.t_compare):
            table.set_heads([tr(f"brain.{b}") for b in BRAINS], [c[f"brain_{b}"] for b in BRAINS])
        self.s_mode.set_texts({m: tr(f"theme.{m}") for m in MODES})
        for r in self.robots:
            r.view.overlay.hint = tr(f"hint.{self.tool or 'view'}")
        self.refresh()

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.LayoutRequest and watched is self.scroll.widget():
            self._fit_panel()
        return super().eventFilter(watched, event)

    def _fit_panel(self) -> None:
        """The panel as wide as its widest row needs (once styled: texts, fonts, language), and at least
        PANEL_WIDTH, beside its scroll bar: nothing of it is cut, nothing is under the bar."""
        need = self.scroll.widget().minimumSizeHint().width() + self.scroll.verticalScrollBar().sizeHint().width()
        width = max(PANEL_WIDTH, need)
        if width != self.scroll.width():
            self.scroll.setFixedWidth(width)

    def _show_height(self) -> None:
        height = self.s_height.value()
        self.l_height.setText(self.texts("label.height", height=height))
        low, high = (round(100 * h) for h in TRAINED_HEIGHT)
        self.l_untrained.setText(self.texts("caption.untrained_height", low=low, high=high))
        self.l_untrained.setVisible(not low <= height <= high)
        for r in self.robots:
            r.view.drop_height = self.drop_height()

    def _show_latency(self) -> None:
        self.l_latency.setText(self.texts("label.latency", latency=self.latency_ms()))

    def latency_ms(self) -> int:
        cfg = self.session.cfg
        return round(1000 * cfg.latency * cfg.control_period)

    # --- Actions (on every robot) ------------------------------------------------------------
    def set_tool(self, name) -> None:
        """Activates the tool `name`, or deactivates it if it is the active one."""
        self._use_tool(None if name == self.tool else name)

    def _use_tool(self, name) -> None:
        self.tool = name
        for n, b in self.tool_buttons.items():
            b.setChecked(n == self.tool)
        self.height_row.setVisible(self.tool == "drop")
        for r in self.robots:
            r.view.set_tool(self.tool)
            r.view.overlay.hint = self.texts(f"hint.{self.tool or 'view'}")

    def throw(self, x: float, y: float, vx: float, vy: float) -> None:
        for r in self.robots:
            r.session.throw(**throw_arguments((x, y), vx, vy))

    def drop_height(self) -> float:
        """(m) Height of a drop: of the bottom of the ball above the plate."""
        return self.s_height.value() / 100

    def drop(self, x: float, y: float, vx: float, vy: float) -> None:
        for r in self.robots:
            r.session.throw(**drop_arguments((x, y), self.drop_height(), vx, vy))

    def push(self, dvx: float, dvy: float) -> None:
        for r in self.robots:
            r.session.push(dvx, dvy)

    def center(self) -> None:
        for r in self.robots:
            r.session.center_ball()

    def set_brain(self, name: str) -> None:
        """The brain of the single robot (side by side, each robot keeps its own)."""
        if not self.split:
            self.session.set_brain(name)

    def set_ball(self, name: str) -> None:
        for r in self.robots:
            r.session.set_ball(name)

    def set_friction(self, name: str) -> None:
        mu = next(mu for mu, n in FRICTION_NAMES.items() if n == name)
        for r in self.robots:
            r.session.set_friction(mu)

    def set_latency(self, periods: int) -> None:
        for r in self.robots:
            r.session.set_latency(periods)
        self._show_latency()

    def set_setpoint(self, name: str) -> None:
        """Center, circle or square, drawn on the plates; "drawn": the tool Draw, and the last path drawn if any."""
        self.draw_row.setVisible(name == "drawn")
        if name == "drawn":
            self._use_tool("draw")
            if self.drawn_path is not None:
                self._set_path(self.drawn_path)
            self._show_drawing()
            return
        if self.tool == "draw":
            self._use_tool(None)
        for r in self.robots:
            r.session.set_setpoint(name)
        f = SETPOINTS[name]
        path = [f(t)[:2] for t in np.linspace(0.0, PATH_DURATION, 161)]
        self._show_path(path if np.ptp(np.array(path), axis=0).max() > 0 else [])

    def _show_path(self, path: list) -> None:
        self.path = path
        for r in self.robots:
            r.view.overlay.path = path

    def draw(self, points: list) -> None:
        """A path drawn on the plate: the setpoint, if it is long enough and stays within DRAW_RADIUS."""
        length = float(np.hypot(*np.diff(np.asarray(points + points[:1]), axis=0).T).sum()) if len(points) > 1 else 0
        if length < MIN_DRAW_LENGTH:
            self._show_drawing("draw.short")
            return
        if max(math.hypot(*p) for p in points) > DRAW_RADIUS:
            self._show_drawing("draw.refused")
            return
        self.s_setpoint.buttons["drawn"].setChecked(True)
        self.draw_row.setVisible(True)
        self._set_path(DrawnPath(tuple(points), self.s_speed.value() / 100))
        self._show_drawing()

    def set_draw_speed(self, speed: int) -> None:
        """(cm/s) The last path drawn, at another speed (a new setpoint, if it is the setpoint)."""
        self.l_speed.setText(self.texts("label.speed", speed=speed))
        if self.drawn_path is not None:
            path = DrawnPath(self.drawn_path.points, speed / 100)
            if self.session.setpoint == "drawn":
                self._set_path(path)
            self.drawn_path = path
            self._show_drawing()

    def _set_path(self, path: DrawnPath) -> None:
        self.drawn_path = path
        for r in self.robots:
            r.session.set_path(path)
        samples = path.samples[::4]
        self._show_path([tuple(p) for p in samples] + [tuple(samples[0])])

    def _show_drawing(self, problem: str = None) -> None:
        """Under the speed: what to do, the path in force, or why a path was not taken."""
        tr, p = self.texts, self.drawn_path
        if problem is not None:
            text = tr(problem, radius=round(100 * DRAW_RADIUS))
        elif p is None:
            text = tr("draw.how")
        else:
            text = tr("draw.path", length=tr.number(100 * p.length, 0), time=tr.number(p.length / p.speed, 1))
        self.l_draw.setText(text + "\n" + tr("draw.untrained"))

    def set_pause(self, paused: bool) -> None:
        self.real_time.paused = paused
        self.b_pause.setText(self.texts("button.resume" if paused else "button.pause"))
        self._update_overlays()
        for r in self.robots:
            r.view.update()

    def set_language(self, code: str) -> None:
        self.texts.set_language(code)
        self.retranslate()

    def set_mode(self, mode: str) -> None:
        """Light or dark theme: the widgets, and the 3D scenes, whose floor continues the background."""
        self.mode = mode
        self.setStyleSheet(stylesheet(mode))
        for r in self.robots:
            if mode != self._scene_mode:
                r.view.set_scene(DisplayScene(r.session.cfg, scene_style(mode)))
            r.view.overlay.colors = COLORS[mode]
            r.view.update()
        self._scene_mode = mode
        self.retranslate()

    def describe_gesture(self, kind: str, start, end, robot: Robot = None) -> tuple:
        """Text and color (key of the theme) of the arrow of a gesture on the plate of `robot` (the first one by
        default): speed, and for a throw its physical limit (good color within it, bad color beyond); for a drop
        with a speed, the angle of arrival from the vertical (good color if the ball lands on the plate, bad color
        if not)."""
        tr, s = self.texts, (robot or self.robots[0]).session
        cfg = s.cfg
        vx, vy = velocity(start, end)
        speed = math.hypot(vx, vy)
        if kind == "push":
            return tr("gesture.push", speed=tr.number(speed, 2)), "accent"
        if kind == "drop":
            if speed < MIN_SPEED:
                return "", "accent"
            landing, angle, _ = drop_flight(start, vx, vy, self.drop_height(), s.phys.data.body("plate").xmat)
            if math.hypot(*landing) > cfg.plate.plate_radius:
                return tr("gesture.drop_off", speed=tr.number(speed, 2)), "bad"
            return tr("gesture.drop", speed=tr.number(speed, 2), angle=tr.number(angle, 0)), "good"
        if speed == 0:
            return "", "accent"
        limit = throw_limit(start, vx, vy, cfg.plate.plate_radius, cfg.theta_max, s.ball_coefficient * G)
        return (tr("gesture.throw", speed=tr.number(speed, 2), limit=tr.number(limit, 2)),
                "good" if speed <= limit else "bad")

    # --- Loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        """One frame: the simulations catch up with real time, then the 3D views (and, from time to time, the
        measurements) are updated."""
        self.real_time.advance(_Together([r.session for r in self.robots]), self.clock())
        for r in self.robots:
            r.follow_fall()
        self.frames += 1
        if self.frames % READOUT_EVERY == 0:
            self.refresh()
        else:
            self._update_overlays()
        for r in self.robots:
            r.view.render_frame()

    def refresh(self) -> None:
        """Measurements, curves, last attempt, comparison and texts over the 3D views."""
        tr, c = self.texts, COLORS[self.mode]
        robots = self._ordered()
        values = {id(r): readouts(r.session) for r in robots}

        def text(v, unit, decimals):
            return "—" if v is None else tr.number(v, decimals) + ("°" if unit == "°" else "")
        if self.split:
            for row, (key, _, unit, decimals) in enumerate(MEASURES):
                for column, r in enumerate(robots):
                    self.t_measures.set_value(row, column, text(values[id(r)][key], unit, decimals))
            self.sparkline.set_series([(distance_history(r.session), c[f"brain_{r.brain}"]) for r in robots])
        else:
            for key, _, unit, decimals in MEASURES:
                self.readouts[key].value.setText(text(values[id(self.robots[0])][key], unit, decimals))
            self.sparkline.set_series([(distance_history(self.session), c["accent"])])
        self.l_last.setText(self.last_attempt_text())
        self._show_comparison()
        self._update_overlays(values)

    def _show_comparison(self) -> None:
        tr, s = self.texts, self.session
        rows = self.past_rows + [row for r in self.robots for row in r.session.summary]
        condition = (s.setpoint, s.path.key if s.path else "", s.ball, s.cfg.mujoco_params.friction, self.latency_ms())
        result = compare(rows, condition, BRAINS)
        for row, (key, _, unit, decimals) in enumerate(INDICATORS):
            for column, brain in enumerate(BRAINS):
                v = result[brain][key]
                self.t_compare.set_value(row, column, "—" if v is None else
                                         tr.number(v, decimals) + (f" {unit}" if unit else ""))
        self.l_compare.setText(tr("caption.compare", setpoint=tr(f"setpoint.{s.setpoint}"), ball=tr(f"ball.{s.ball}"),
                                  mu=tr.number(s.cfg.mujoco_params.friction, 2), latency=self.latency_ms()))

    def _update_overlays(self, values: dict = None) -> None:
        tr, rt, c = self.texts, self.real_time, COLORS[self.mode]
        clock = tr("overlay.clock", time=tr.number(self.session.t, 1), speed=tr.number(rt.speed, 2),
                   fps=tr.number(rt.frames_per_second, 0))
        for r in self.robots:
            s, o = r.session, r.view.overlay
            state = tr("status.paused") if rt.paused else self.state_text(r, short=self.split)
            o.state = tr("state.robot", brain=tr(f"brain.{r.brain}"), state=state) if self.split else state
            o.state_color = c["text3"] if rt.paused else c["bad"] if s.state == "fallen" else c["good"]
            o.bottom_right = clock
            o.setpoint = tuple(s.sim.cfg.trajectory(s.t)[:2])
            if values is not None:
                v = values[id(r)]
                o.bottom_left = ("" if v["x_mm"] is None else
                                 tr("overlay.position", x=tr.number(v["x_mm"], 1), y=tr.number(v["y_mm"], 1)))

    def state_text(self, robot: Robot = None, short: bool = False) -> str:
        """What the robot (the first one by default) is doing; short: without the name of its brain."""
        tr = self.texts
        robot = robot or self.robots[0]
        s = robot.session
        if s.state == "fallen":
            return tr("state.fallen", time=tr.number(robot.fall_left(), 1))
        if s.attempt is not None:
            a = s.attempt
            if short:
                return tr("state.attempt_short", action=tr(f"action.{a['action']}"))
            return tr("state.attempt", action=tr(f"action.{a['action']}"), brain=tr(f"brain.{a['brain']}"))
        return tr("state.ready")

    def last_attempt_text(self) -> str:
        """Result of the last finished attempt of each robot, then its settings."""
        tr = self.texts
        rows = [r.session.summary[-1] for r in self._ordered() if r.session.summary]
        if not rows:
            return tr("last.none")
        lines = []
        for row in rows:
            duration = row["stabilized_in_s"] if row["result"] == "stabilized" else row["duration_s"]
            result = tr("result." + row["result"].replace(" ", "_"), time=tr.number(duration, 2))
            lines.append(tr("last.row", action=tr(f"action.{row['action']}"), brain=tr(f"brain.{row['brain']}"),
                            result=result))
        row = rows[-1]
        lines.append(tr("last.settings", ball=tr(f"ball.{row['ball']}"), mu=tr.number(row["friction"], 2),
                        latency=row["latency_ms"]))
        return "\n".join(lines)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        for r in self.robots:
            r.view.close_renderer()
            r.session.close()
        super().closeEvent(event)
