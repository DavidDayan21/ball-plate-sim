"""Main window of the software: the control panel on the left, the robot in 3D on the right.

A single loop, in the thread of the interface: a timer advances the simulation by the real time elapsed
(simulation.session.RealTime, up to MAX_LAG per frame), then renders the 3D view. The panel acts on the session directly, between two frames.
Its measurements are refreshed every READOUT_EVERY frames (readable, and cheaper).
"""
import time
from dataclasses import replace

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QMainWindow, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from app.i18n import LANGUAGES, Texts
from app.readouts import distance_history, readouts
from app.scene import DisplayScene
from app.theme import COLORS, MODES, load_fonts, scene_style, stylesheet, system_mode
from app.view3d import View3D
from app.widgets import Card, Readout, Segmented, Sparkline, label
from simulation.session import RealTime, Session, sandbox_config

FRAMES_PER_SECOND = 60
MAX_LAG = 0.1           # (s) of real time caught up per frame at most: the simulation stays in step with real time as
                        # long as a frame takes less (measured: up to 45 ms on an integrated graphics card)
READOUT_EVERY = 5                                           # frames
BRAINS = ("pid", "imitation")                               # the PID brain, the AI brain (brains/imitation.npz)
THROW = dict(distance=0.10, angle_deg=180, speed=0.4)       # toward the center: the default throw of the sandbox
DROP = dict(distance=0.0, angle_deg=0.0, height=0.305)      # above the center, from the height of the ITTF rule
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
        self.view = View3D(DisplayScene(self.session.cfg, scene_style(self.mode)), self.session.phys.data,
                           self.session.cfg.geometry.height - 0.035)
        self._scene_mode = self.mode                        # theme of the display scene
        self.frames = 0
        self._build()
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
        self.b_throw = QPushButton(clicked=self.throw)
        self.b_drop = QPushButton(clicked=self.drop)
        self.b_center = QPushButton(clicked=self.center)
        self.b_pause = QPushButton(checkable=True, toggled=self.set_pause, objectName="ghost")
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, b in enumerate((self.b_throw, self.b_drop, self.b_center, self.b_pause)):
            grid.addWidget(b, i // 2, i % 2)
        self.c_ball.body.addLayout(grid)

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
        for w in (self.c_brain, self.c_ball, self.c_measures, self.c_last):
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
        for b, key in ((self.b_throw, "throw"), (self.b_drop, "drop"), (self.b_center, "center")):
            b.setText(tr(f"button.{key}"))
            b.setToolTip(tr(f"tip.{key}"))
        self.b_pause.setText(tr("button.resume" if self.real_time.paused else "button.pause"))
        self.c_measures.set_title(tr("section.measures"))
        self.c_measures.setToolTip(tr("tip.measures"))
        for key, name, unit, _ in MEASURES:
            self.readouts[key].set_name(tr(f"measure.{name}").upper(), "" if unit == "°" else unit)  # 12,3°
        self.l_history.setText(tr("caption.history"))
        self.c_last.set_title(tr("section.last"))
        self.s_mode.set_texts({m: tr(f"theme.{m}") for m in MODES})
        self.view.overlay.hint = tr("view.hint")
        self.refresh()

    # --- Actions -----------------------------------------------------------------------------
    def throw(self) -> None:
        self.session.throw(**THROW)

    def drop(self) -> None:
        self.session.throw(**DROP)

    def center(self) -> None:
        self.session.center_ball()

    def set_brain(self, name: str) -> None:
        self.session.set_brain(name)

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

    # --- Loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        """One frame: the simulation catches up with real time, then the 3D view (and, from time to time, the
        measurements) are updated."""
        self.real_time.advance(self.session, self.clock())
        self.frames += 1
        if self.frames % READOUT_EVERY == 0:
            self.refresh()
        else:
            self._update_overlay()
        self.view.render_frame()

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
        if values is not None:
            o.bottom_left = ("" if values["x_mm"] is None else
                             tr("overlay.position", x=tr.number(values["x_mm"], 1), y=tr.number(values["y_mm"], 1)))

    def state_text(self) -> str:
        """What the robot is doing."""
        tr, s = self.texts, self.session
        if s.state == "fallen":
            return tr("state.fallen")
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
        return tr("last.row", action=tr(f"action.{row['action']}"), brain=tr(f"brain.{row['brain']}"), result=result)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.view.close_renderer()
        self.session.close()
        super().closeEvent(event)
