"""Main window of the software: the robot in 3D and its control panel.

A single loop, in the thread of the interface: a timer advances the simulation by the real time elapsed
(simulation.session.RealTime), then renders the 3D view. The panel acts on the session directly, between two frames.
"""
import time
from dataclasses import replace

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QRadioButton, QVBoxLayout, QWidget)

from app.i18n import LANGUAGES, Texts
from app.view3d import View3D
from simulation.session import RealTime, Session, sandbox_config

FRAMES_PER_SECOND = 60
BRAINS = ("pid", "imitation")                               # the PID brain, the AI brain (brains/imitation.npz)
THROW = dict(distance=0.10, angle_deg=180, speed=0.4)       # toward the center: the default throw of the sandbox
DROP = dict(distance=0.0, angle_deg=0.0, height=0.305)      # above the center, from the height of the ITTF rule
PANEL_WIDTH = 230                                           # (pixels)


class MainWindow(QMainWindow):
    def __init__(self, texts: Texts, brain: str = "pid", clock=time.perf_counter):
        """clock: real time (s), at every frame (replaced by a simulated clock in the tests)."""
        super().__init__()
        self.texts = texts
        self.clock = clock
        self.session = Session(replace(sandbox_config(), brain=brain), record=False)
        self.real_time = RealTime()
        self.view = View3D(self.session.phys.model, self.session.phys.data, self.session.cfg.geometry.height)
        self._build()
        self.retranslate()
        self.timer = QTimer(self, timerType=Qt.PreciseTimer, interval=round(1000 / FRAMES_PER_SECOND))
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # --- Layout ------------------------------------------------------------------------------
    def _build(self) -> None:
        self.b_throw = QPushButton(clicked=self.throw)
        self.b_drop = QPushButton(clicked=self.drop)
        self.b_center = QPushButton(clicked=self.center)
        self.g_ball = QGroupBox()
        ball = QVBoxLayout(self.g_ball)
        for b in (self.b_throw, self.b_drop, self.b_center):
            ball.addWidget(b)

        self.g_brain = QGroupBox()
        brain = QVBoxLayout(self.g_brain)
        self.brain_group = QButtonGroup(self)
        self.brain_buttons = {}
        for name in BRAINS:
            b = QRadioButton(checked=name == self.session.cfg.brain)
            b.toggled.connect(lambda on, n=name: on and self.set_brain(n))
            self.brain_group.addButton(b)
            brain.addWidget(b)
            self.brain_buttons[name] = b

        self.b_pause = QPushButton(checkable=True, toggled=self.set_pause)
        self.l_language = QLabel()
        self.c_language = QComboBox()
        for code, name in LANGUAGES.items():
            self.c_language.addItem(name, code)
        self.c_language.setCurrentIndex(self.c_language.findData(self.texts.code))
        self.c_language.currentIndexChanged.connect(lambda _: self.set_language(self.c_language.currentData()))
        self.l_hint = QLabel(wordWrap=True)
        self.l_hint.setStyleSheet("color: gray;")

        panel = QWidget()
        panel.setFixedWidth(PANEL_WIDTH)
        column = QVBoxLayout(panel)
        for w in (self.g_ball, self.g_brain, self.b_pause):
            column.addWidget(w)
        column.addStretch(1)
        for w in (self.l_hint, self.l_language, self.c_language):
            column.addWidget(w)

        central = QWidget()
        row = QHBoxLayout(central)
        row.addWidget(self.view, 1)
        row.addWidget(panel)
        self.setCentralWidget(central)

        self.l_state = QLabel()
        self.l_clock = QLabel()
        self.statusBar().addWidget(self.l_state, 1)
        self.statusBar().addPermanentWidget(self.l_clock)

    def retranslate(self) -> None:
        """Writes all the texts in the current language."""
        tr = self.texts
        self.setWindowTitle(tr("window.title"))
        self.g_ball.setTitle(tr("group.ball"))
        for b, key in ((self.b_throw, "throw"), (self.b_drop, "drop"), (self.b_center, "center")):
            b.setText(tr(f"button.{key}"))
            b.setToolTip(tr(f"tip.{key}"))
        self.g_brain.setTitle(tr("group.brain"))
        self.g_brain.setToolTip(tr("tip.brain"))
        for name, b in self.brain_buttons.items():
            b.setText(tr(f"brain.{name}"))
        self.b_pause.setText(tr("button.resume" if self.real_time.paused else "button.pause"))
        self.l_language.setText(tr("label.language"))
        self.l_hint.setText(tr("view.hint"))
        self._show_state()

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

    def set_language(self, code: str) -> None:
        self.texts.set_language(code)
        self.retranslate()

    # --- Loop --------------------------------------------------------------------------------
    def tick(self) -> None:
        """One frame: the simulation catches up with real time, then the 3D view and the texts are updated."""
        self.real_time.advance(self.session, self.clock())
        self.view.render_frame()
        self._show_state()

    def _show_state(self) -> None:
        tr, s, rt = self.texts, self.session, self.real_time
        clock = tr("status.clock", time=tr.number(s.t, 1), speed=tr.number(rt.speed, 2),
                   fps=tr.number(rt.frames_per_second, 0))
        self.l_clock.setText(f"{tr('status.paused')} · {clock}" if rt.paused else clock)
        self.l_state.setText(self.state_text())

    def state_text(self) -> str:
        """What the robot is doing, or the result of the last attempt."""
        tr, s = self.texts, self.session
        if s.state == "fallen":
            return tr("state.fallen")
        if s.attempt is not None:
            a = s.attempt
            return tr("state.attempt", action=tr(f"action.{a['action']}"), brain=tr(f"brain.{a['brain']}"))
        if s.summary:
            row = s.summary[-1]
            duration = row["stabilized_in_s"] if row["result"] == "stabilized" else row["duration_s"]
            result = tr("result." + row["result"].replace(" ", "_"), time=tr.number(duration, 2))
            return tr("state.last", action=tr(f"action.{row['action']}"), brain=tr(f"brain.{row['brain']}"),
                      result=result)
        return tr("state.ready")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.view.close_renderer()
        self.session.close()
        super().closeEvent(event)
