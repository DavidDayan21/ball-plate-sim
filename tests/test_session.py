"""Sandbox session, without display: actions, end of attempts, files written."""
import csv
import math
from dataclasses import replace

import pytest

import numpy as np

from controller.plate import PlateController
from controller.policy import PolicyController
from simulation.mujoco_physics import MujocoPhysics
from simulation.session import BALLS, RealTime, Session, sandbox_config


def advance(s: Session, duration: float) -> None:
    for _ in range(round(duration / s.dt)):
        s.step()


@pytest.fixture
def session(tmp_path):
    s = Session(folder=tmp_path)
    advance(s, 1.0)
    return s


def test_dropped_ball_bounces_then_stabilizes(session):
    session.throw(0.0, 0.0, height=0.305)                   # drop from 30.5 cm at the center
    advance(session, 15.0)
    l = session.summary[-1]
    assert l["action"] == "drop" and l["result"] == "stabilized"
    assert l["landings"] >= 5
    # First bounce: a little below e² · h = 0.88² × 305 mm (the plate, mounted on motors, absorbs some energy)
    assert 150 < l["max_bounce_mm"] < 0.91 ** 2 * 305


def test_throw_caught_then_too_fast_falls(session):
    session.throw(0.10, 180, speed=0.4)
    advance(session, 15.0)
    assert session.summary[-1]["result"] == "stabilized"
    session.throw(0.10, 180, speed=1.0)
    advance(session, 3.0)
    assert session.summary[-1]["result"] == "fallen" and session.state == "fallen"
    advance(session, 2.0)                                     # the simulation goes on without a ball
    session.center_ball()                                     # and restarts
    advance(session, 5.0)
    assert session.summary[-1]["action"] == "center" and session.summary[-1]["result"] == "stabilized"


def test_disturbance_detected_automatically(session):
    session.center_ball()
    advance(session, 5.0)
    assert session.state == "waiting"
    session.phys.push(0.3, 0.0)                            # like a push with the mouse
    advance(session, 1.0)
    assert session.state == "attempt" and session.attempt["action"] == "disturbance"
    advance(session, 10.0)
    assert session.summary[-1]["action"] == "disturbance"
    assert session.summary[-1]["result"] == "stabilized"


def test_files_written(session):
    session.throw(0.10, 180, speed=0.4)
    advance(session, 2.0)
    session.set_setpoint("circle")
    session.free_attempt()                                 # interrupts the throw
    advance(session, 2.0)
    summary = session.close()
    assert [l["result"] for l in summary] == ["interrupted", "interrupted"]

    with open(session.folder / "summary.csv", encoding="utf-8") as f:
        assert f.read(1) != "﻿"                      # UTF-8 without BOM
        f.seek(0)
        rows = list(csv.DictReader(f, delimiter=","))
    assert len(rows) == 2 and rows[0]["action"] == "throw" and rows[1]["setpoint"] == "circle"
    assert rows[0]["brain"] == "pid"
    assert rows[0]["speed_m_s"] == "0.400"                # decimal point, same number of decimals
    assert rows[0]["stabilized_in_s"] == ""               # missing value: empty cell

    with open(session.folder / "attempt_002.csv", encoding="utf-8") as f:
        measurements = list(csv.DictReader(f, delimiter=","))
    assert len(measurements) == 100                                # 2 s at 50 Hz
    # The circle setpoint starts at its starting point: 6 cm at 45°
    x, y = float(measurements[0]["x_setpoint_mm"]), float(measurements[0]["y_setpoint_mm"])
    assert math.isclose(x, 60 * math.cos(math.pi / 4), abs_tol=0.01) and math.isclose(y, x, abs_tol=0.01)


def test_session_with_the_ai_brain(tmp_path):
    # The sandbox driven by the learned policy (python sandbox.py --brain imitation)
    s = Session(replace(sandbox_config(), brain="imitation"), folder=tmp_path)
    assert isinstance(s.sim.brain, PolicyController)
    advance(s, 1.0)
    s.throw(0.10, 180, speed=0.4)
    advance(s, 4.0)
    s.set_latency(3)                                       # the brain is created again, still the AI one
    assert isinstance(s.sim.brain, PolicyController) and s.sim.brain.latency == 3
    advance(s, 1.0)
    summary = s.close()
    assert summary[0]["result"] == "stabilized" and math.hypot(*s.sim.phys.ball()) < 0.005


def test_brain_changed_between_attempts(tmp_path):
    s = Session(folder=tmp_path)
    advance(s, 1.0)
    s.throw(0.10, 180, speed=0.4)
    advance(s, 1.0)
    s.set_brain("pid")                                     # the same brain: nothing changes
    assert s.attempt["action"] == "throw"
    s.set_brain("imitation")                               # an attempt has a single brain: the throw ends
    assert isinstance(s.sim.brain, PolicyController) and s.attempt is None
    assert s.summary[-1]["brain"] == "pid" and s.summary[-1]["result"] == "interrupted"
    advance(s, 5.0)                                        # the AI brain takes the ball over, from zero
    assert s.state != "fallen"
    s.throw(0.10, 180, speed=0.4)
    advance(s, 6.0)
    s.set_brain("pid")
    assert isinstance(s.sim.brain, PlateController)
    assert all(l["brain"] == "imitation" for l in s.summary[1:])
    assert s.summary[-1]["action"] == "throw" and s.summary[-1]["result"] == "stabilized"


def test_settings_end_the_attempt_and_are_recorded(tmp_path):
    s = Session(folder=tmp_path, record=False)
    r, m = s.cfg.plate.ball_radius, s.cfg.mujoco_params.ball_mass
    assert s.ball == "ping_pong" and s.ball_coefficient == pytest.approx(0.6)
    s.throw(0.05, 180, speed=0.1)
    s.set_ball("ping_pong")                                # the same ball: nothing changes
    assert s.attempt is not None
    s.set_ball("solid")                                    # an attempt has a single ball: it ends
    assert s.attempt is None and s.summary[-1]["result"] == "interrupted" and s.summary[-1]["ball"] == "ping_pong"
    ball = s.phys.model.body("ball").id
    assert s.phys.model.body_inertia[ball] == pytest.approx([0.4 * m * r * r] * 3)
    assert s.ball_coefficient == pytest.approx(1 / 1.4)
    s.throw(0.05, 180, speed=0.1)
    s.set_friction(0.2)
    assert s.attempt is None and s.summary[-1]["ball"] == "solid" and s.summary[-1]["friction"] == 1.0
    for name in ("ball", "plate"):
        assert s.phys.model.geom_friction[s.phys.model.geom(name).id, 0] == 0.2
    s.throw(0.05, 180, speed=0.1)
    s.set_latency(3)
    assert s.attempt is None and s.summary[-1]["friction"] == 0.2 and s.summary[-1]["latency_ms"] == 20
    s.set_ball("heavy")
    assert s.phys.model.body_mass[ball] == pytest.approx(10 * m)
    s.throw(0.05, 180, speed=0.1)
    advance(s, 4.0)
    row = s.summary[-1]
    assert (row["ball"], row["friction"], row["latency_ms"], row["result"]) == ("heavy", 0.2, 60, "stabilized")


def test_ball_and_friction_changed_in_place_as_if_built():
    """A ball and a friction changed in the model in place behave exactly as in a model built with them (measured:
    the same model constants, and the same motion to the last bit over 1 s of bounces and rolling)."""
    base = sandbox_config()
    for name, k in (("solid", 2 / 5), ("heavy", base.plate.k_inertia)):
        built = MujocoPhysics(replace(base, mujoco_params=replace(base.mujoco_params, friction=0.02, **BALLS[name])))
        changed = MujocoPhysics(base)
        changed.set_ball(BALLS[name]["ball_mass"], k)
        changed.set_friction(0.02)
        for field in ("body_mass", "body_inertia", "body_subtreemass", "body_invweight0", "dof_invweight0",
                      "geom_friction"):
            assert np.array_equal(getattr(built.model, field), getattr(changed.model, field)), (name, field)
        for p in (built, changed):
            p.place_ball(0.05, -0.02, 0.10, -0.3, 0.2)
        for _ in range(2000):
            for p in (built, changed):
                p.command(0.02, -0.01)
                p.step(base.dt_physics)
        assert np.array_equal(built.data.qpos, changed.data.qpos)


class Clockwork:
    """Stands for a session: a time that advances by steps of 0.5 ms."""
    dt = 0.0005

    def __init__(self):
        self.n = 0

    @property
    def t(self) -> float:
        return self.n * self.dt

    def step(self) -> None:
        self.n += 1


def test_real_time():
    s, clock = Clockwork(), RealTime(max_lag=0.05)
    clock.advance(s, 10.0)                                 # starts the clock: no step
    assert s.n == 0
    for i in range(1, 181):                                # 3 s of frames, 60 per second
        clock.advance(s, 10.0 + i / 60)
    assert abs(s.t - 3.0) < s.dt                           # in step with real time, without drifting
    assert clock.speed == pytest.approx(1.0, abs=0.001)
    assert clock.frames_per_second == pytest.approx(60.0, abs=0.1)
    clock.advance(s, 14.0)                                 # a frame 1 s late: only 50 ms are caught up
    assert abs(s.t - 3.05) < s.dt
    clock.paused = True
    t = s.t
    clock.advance(s, 14.02)                                # in pause, the simulation waits
    assert s.t == t


def test_push(tmp_path):
    s = Session(folder=tmp_path)
    s.center_ball()
    advance(s, 3.0)
    s.push(0.3, 0.0)
    assert s.attempt["action"] == "push" and s.attempt["params"] == dict(dvx_m_s=0.3, dvy_m_s=0.0)
    assert s.phys.ball_velocity()[0] == pytest.approx(0.3, abs=0.02)
    advance(s, 6.0)
    assert s.summary[-1]["action"] == "push" and s.summary[-1]["result"] == "stabilized"
    s.throw(0.10, 180, speed=1.0)                          # too fast: falls
    advance(s, 3.0)
    assert s.state == "fallen"
    s.push(0.3, 0.0)                                       # a fallen ball is not pushed
    assert s.attempt is None and s.state == "fallen"


def test_the_ground_stops_the_fallen_ball(tmp_path):
    s = Session(folder=tmp_path)
    s.throw(0.10, 180, speed=1.2)                          # falls beyond the edge, lands on the ground
    advance(s, 3.0)
    ground = s.phys.model.geom("ground").id
    z = s.phys.data.body("ball").xpos[2]
    assert s.state == "fallen"
    assert z == pytest.approx(s.phys.model.geom_pos[ground][2] + s.cfg.plate.ball_radius, abs=0.03)
