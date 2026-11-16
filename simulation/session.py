"""Sandbox session: a series of attempts on the same MuJoCo robot, which never stops.

An attempt starts with an action (throw, drop, center, free attempt) or, while waiting,
as soon as the ball moves more than 2 cm away from the setpoint (pushed with the mouse, for example).
When the ball is moved (throw, drop, center), the sensor and the controller restart from zero
(integral, predictor, latency): the attempts are independent of each other. As at the start of a
simulation, the first image then arrives one latency later; until then the plate keeps its command.
It ends when:
    - the ball falls;
    - the ball stays within 1 cm of the setpoint, resting on the plate, for 2 s (stabilized);
    - it lasts longer than max_duration (not stabilized);
    - another attempt starts, or a setting is changed (brain, ball, friction, latency, compensation): an attempt
      has a single setting of each (interrupted).

RealTime keeps a live session in step with real time (the sandbox, the software of app/).

Data written to results/attempts/<date_time>/:
    attempt_001.csv ...  all the measurements of the attempt, at 50 Hz
    summary.csv          one row per attempt (parameters and result)
International CSV format: "," separator, decimal point, UTF-8 encoding (no BOM).
"""
from collections import deque
from dataclasses import fields, replace
import csv
import datetime
import math
from pathlib import Path

import numpy as np

from simulation.loop import Results, Simulation
from simulation.config import Config, MujocoParams, square, center, circle

SETPOINTS = {"center": center, "circle": circle, "square": square}
# The balls that can be chosen (same radius): the ping-pong ball of the model, a solid one, a heavy one, as in the
# robustness variants of the campaign. All within what the AI brain saw in training (k from 2/5 to 2/3, 1 to 10
# times the mass of the ping-pong ball). k_actual None: the k of the controller, 2/3.
BALLS = {"ping_pong": dict(ball_mass=MujocoParams.ball_mass, k_actual=None),
         "solid": dict(ball_mass=MujocoParams.ball_mass, k_actual=2 / 5),
         "heavy": dict(ball_mass=10 * MujocoParams.ball_mass, k_actual=None)}
FRICTIONS = (1.0, 0.2, 0.02)    # ball-plate μ that can be chosen, as in the campaign (training: 0.02 to 1)
STABLE_THRESHOLD = 0.01         # (m) distance to the setpoint
STABLE_DURATION = 2.0           # (s)
DISTURBANCE_THRESHOLD = 0.02    # (m) while waiting: beyond this, an attempt starts by itself
PRE_DISTURBANCE = 0.5           # (s) of measurements kept before the trigger
MIN_FLIGHT = 0.010              # (s) a shorter flight is not counted as a landing
RESTING_HEIGHT = 0.001          # (m) the ball is considered resting below this height


def sandbox_config() -> Config:
    """Complete robot, ball at rest at the center, setpoint at the center."""
    return Config(trajectory=center, start_position=(0.0, 0.0), duration=math.inf)


def _shift(function, t0: float):
    """Setpoint that starts at t0 (a circle starts at its starting point)."""
    return lambda t: function(t - t0)


def _keep_last(r: Results, n: int) -> Results:
    """Copy of the last n samples of r."""
    tail = Results()
    for f in fields(Results):
        value = getattr(r, f.name)
        if isinstance(value, list):
            setattr(tail, f.name, value[-n:] if n > 0 else [])
    return tail


def _format_number(v: float, decimals: int = 3) -> str:
    """Number with a fixed number of decimals (decimal point); empty cell if missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{decimals}f}"


class Session:
    def __init__(self, cfg: Config = None, folder="results/attempts", max_duration: float = 30.0,
                 ground: bool = True, record: bool = True):
        self.cfg = cfg or sandbox_config()
        if self.cfg.physics != "mujoco":
            raise ValueError("The sandbox uses the MuJoCo physics")
        self.sim = Simulation(self.cfg)
        self.phys = self.sim.phys
        self.dt = self.cfg.dt_physics
        self.ppc = self.sim.steps_per_control
        self.max_duration = max_duration
        self.setpoint = "center"
        self.t_setpoint = 0.0
        self._apply_config()
        pm = self.cfg.mujoco_params
        self.ball = next((name for name, b in BALLS.items()
                          if all(getattr(pm, key) == value for key, value in b.items())), "other")
        if ground:
            self._enable_ground()
        # Recording
        self.record = record
        self.folder = Path(folder) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.summary = []                # one row (dict) per finished attempt
        self.number = 0
        self.attempt = None              # dict of the current attempt, None while waiting
        self.state = "waiting"           # "attempt", "waiting" or "fallen"
        self.message = "Ready."
        # Live display, 10 s of history: (t, distance to the setpoint, θx, θy, height, x, y,
        # x_setpoint, y_setpoint)
        self.curves = deque(maxlen=round(10.0 / self.cfg.control_period))
        # Step-by-step tracking (landings)
        self._steps_in_flight = 0

    # --- Settings ----------------------------------------------------------------------------
    def _enable_ground(self) -> None:
        """The ground stops the fallen ball (without it, the ball would fall forever).
        Collision bit 2: the ground only touches the ball, never the plate. MuJoCo first sorts the pairs of bodies
        with masks of the bodies (body_contype, body_conaffinity: the union of those of their geoms, computed when
        the model is compiled): they are updated too, otherwise the pair world-ball is discarded before the geoms
        are even compared, and the ball falls through the ground."""
        m = self.phys.model
        ground, ball = m.geom("ground").id, m.geom("ball").id
        m.geom_contype[ground] = 2
        m.geom_conaffinity[ground] = 2
        m.geom_conaffinity[ball] |= 2
        m.body_contype[m.geom_bodyid[ground]] |= 2
        m.body_conaffinity[m.geom_bodyid[ground]] |= 2
        m.body_conaffinity[m.geom_bodyid[ball]] |= 2

    def _apply_config(self, **changes) -> None:
        self.cfg = replace(self.cfg, **changes)
        self.sim.cfg = replace(self.cfg, trajectory=_shift(SETPOINTS[self.setpoint], self.t_setpoint))

    def set_setpoint(self, name: str) -> None:
        self.setpoint, self.t_setpoint = name, self.sim.t
        self._apply_config()

    # A change of setting ends the current attempt (interrupted): an attempt has a single setting of each. The ball
    # is not touched.
    def _interrupt(self) -> None:
        if self.attempt is not None:
            self._finish("interrupted")

    def set_latency(self, periods: int) -> None:
        """Measurement latency in control periods (1 = 20 ms). Resets the controller."""
        if periods == self.cfg.latency:
            return
        self._interrupt()
        self._apply_config(latency=periods)
        self.sim.reset_control()

    def set_compensation(self, active: bool) -> None:
        if active == self.cfg.latency_compensation:
            return
        self._interrupt()
        self._apply_config(latency_compensation=active)
        self.sim.reset_control()

    def set_brain(self, name: str) -> None:
        """Brain of the robot: "pid" or a learned policy of brains/ ("imitation"). The new brain starts from zero,
        like after a throw."""
        if name == self.cfg.brain:
            return
        self._interrupt()
        self._apply_config(brain=name)
        self.sim.reset_control()

    def set_ball(self, name: str) -> None:
        """Ball of BALLS, in place of the one in play (it keeps its place and velocity). The brain is not told: the
        PID estimates the gain c·g of the ball on line, the AI brain saw these balls in training."""
        if name == self.ball:
            return
        self._interrupt()
        self.ball = name
        self._apply_config(mujoco_params=replace(self.cfg.mujoco_params, **BALLS[name]))
        self.phys.set_ball(self.cfg.mujoco_params.ball_mass, self.k_ball)

    def set_friction(self, friction: float) -> None:
        """Ball-plate friction μ."""
        if friction == self.cfg.mujoco_params.friction:
            return
        self._interrupt()
        self._apply_config(mujoco_params=replace(self.cfg.mujoco_params, friction=friction))
        self.phys.set_friction(friction)

    @property
    def k_ball(self) -> float:
        """k of the ball in play, J = k·m·r²."""
        k = self.cfg.mujoco_params.k_actual
        return self.cfg.plate.k_inertia if k is None else k

    @property
    def ball_coefficient(self) -> float:
        """c = 1/(1+k) of the ball in play, in ẍ = c·g·sin θ."""
        return 1.0 / (1.0 + self.k_ball)

    def enable_camera(self) -> None:
        """Measurement by the virtual camera, to be called from the thread that advances the simulation."""
        if self.sim.camera is None:
            self.sim.create_camera()
            self.sim.reset_control()

    def close_camera(self) -> None:
        """Releases the camera (from the thread that created it); back to the synthetic sensor."""
        if self.sim.camera is not None:
            self.sim.camera.close()
            self.sim.camera = None
            self.sim.reset_control()

    @property
    def sensor(self) -> str:
        return "camera" if self.sim.camera is not None else "synthetic"

    @property
    def t(self) -> float:
        return self.sim.t

    # --- Actions -----------------------------------------------------------------------------
    def throw(self, distance: float, angle_deg: float, speed: float = 0.0, height: float = 0.0,
               deviation_deg: float = 0.0, action: str = None) -> None:
        """Places the ball at the starting point (distance, angle) from the center, at `height` above the
        plate, with a horizontal velocity directed toward the center, rotated by `deviation_deg`.
        speed = 0: simple drop. height = 0: ball thrown rolling."""
        a = math.radians(angle_deg)
        x, y = distance * math.cos(a), distance * math.sin(a)
        aim = a + math.pi + math.radians(deviation_deg)
        vx, vy = speed * math.cos(aim), speed * math.sin(aim)
        action = action or ("drop" if speed == 0 else "throw")
        self._new_attempt(action, dict(distance_cm=100 * distance, angle_deg=angle_deg,
                                              speed_m_s=speed, height_cm=100 * height,
                                              deviation_deg=deviation_deg), reset=True)
        self.phys.place_ball(x, y, height, vx, vy, 0.0)
        self.phys.read_period_airborne_steps()          # resets the period counter
        self.sim.last_measurement = (x, y, height)

    def center_ball(self) -> None:
        """Ball placed motionless at the center."""
        self.throw(0.0, 0.0, action="center")

    def push(self, dvx: float, dvy: float) -> None:
        """Pushes the ball: adds (Δvx, Δvy) (m/s, horizontal) to its velocity, with the matching rolling. A new
        attempt starts; the brain keeps its memory, as for a push of the hand."""
        if self.state == "fallen":
            self.message = "The ball has fallen: use Throw, Drop or Center."
            return
        self._new_attempt("push", dict(dvx_m_s=dvx, dvy_m_s=dvy))
        self.phys.push(dvx, dvy)

    def free_attempt(self) -> None:
        """Records without touching the ball (to push it with the mouse, change the setpoint...)."""
        if self.state == "fallen":
            self.message = "The ball has fallen: use Throw, Drop or Center."
            return
        self._new_attempt("free", {})

    # --- Attempts ----------------------------------------------------------------------------
    def _new_attempt(self, action: str, params: dict, before: Results = None,
                            reset: bool = False) -> None:
        if self.attempt is not None:
            self._finish("interrupted")
        self.number += 1
        if reset:
            self.sim.reset_control()
        self.sim.r = before if before is not None else Results()
        self.attempt = dict(number=self.number, action=action, params=params, t0=self.t,
                              time=datetime.datetime.now().strftime("%H:%M:%S"),
                              setpoint=self.setpoint, brain=self.cfg.brain, sensor=self.sensor, ball=self.ball,
                              friction=self.cfg.mujoco_params.friction,
                              latency_ms=round(1000 * self.cfg.latency * self.cfg.control_period),
                              compensation=self.cfg.latency_compensation,
                              stable_since=None, landings=0, max_bounce=0.0)
        self.state = "attempt"
        self._steps_in_flight = 0
        self.message = f"Attempt {self.number} ({action}) in progress."

    def _finish(self, result: str) -> dict:
        att = self.attempt
        r = self.sim.r
        t0 = att["t0"]
        after = [i for i, t in enumerate(r.t) if t >= t0]
        dist = [math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) for i in after]
        edge = [math.hypot(r.x[i], r.y[i]) for i in after]
        tilt = [max(abs(r.theta_x[i]), abs(r.theta_y[i])) for i in after]
        torques = [max(abs(c) for c in r.torques[i]) for i in after if r.torques and r.torques[i]]
        airborne = [r.airborne[i] for i in after] if r.airborne else []
        row = dict(
            number=att["number"], time=att["time"], action=att["action"], **att["params"],
            setpoint=att["setpoint"], brain=att["brain"], sensor=att["sensor"], ball=att["ball"],
            friction=att["friction"], latency_ms=att["latency_ms"],
            compensation=att["compensation"],
            result=result, duration_s=self.t - t0,
            stabilized_in_s=(att["stable_since"] - t0) if result == "stabilized" else None,
            max_error_mm=1000 * max(dist) if dist else None,
            min_edge_margin_mm=1000 * (self.cfg.plate.plate_radius - max(edge)) if edge else None,
            landings=att["landings"], max_bounce_mm=1000 * att["max_bounce"],
            max_tilt_deg=math.degrees(max(tilt)) if tilt else None,
            max_torque_Nm=float(max(torques)) if torques else None,
            airborne_pct=100 * float(np.mean(airborne)) if airborne else None,
            file=f"attempt_{att['number']:03d}.csv")
        self.summary.append(row)
        if self.record:
            self._write(r, row)
        self.attempt = None
        self.state = "fallen" if result == "fallen" else "waiting"
        self.message = self._describe(row)
        self.sim.r = Results()
        return row

    @staticmethod
    def _describe(l: dict) -> str:
        result = "fell" if l["result"] == "fallen" else l["result"]   # display only: "fallen" stays in the data
        text = f"Attempt {l['number']} ({l['action']}): {result}"
        if l["stabilized_in_s"] is not None:
            text += f" in {l['stabilized_in_s']:.2f} s"
        else:
            text += f" after {l['duration_s']:.2f} s"
        if l["landings"]:
            plural = "" if l["landings"] == 1 else "s"
            text += f", {l['landings']} landing{plural}, max bounce {l['max_bounce_mm']:.0f} mm"
        if l["min_edge_margin_mm"] is not None and l["result"] != "fallen":
            text += f", came within {l['min_edge_margin_mm']:.0f} mm of the edge"
        return text + "."

    # --- Files -------------------------------------------------------------------------------
    COLUMNS = ["t_s", "x_mm", "y_mm", "height_mm", "vx_m_s", "vy_m_s", "x_setpoint_mm", "y_setpoint_mm",
                "setpoint_distance_mm", "theta_x_command_deg", "theta_y_command_deg", "theta_x_actual_deg",
                "theta_y_actual_deg", "motor1_deg", "motor2_deg", "motor3_deg", "torque1_Nm", "torque2_Nm",
                "torque3_Nm", "airborne_pct", "measurement_error_mm"]

    def _write(self, r: Results, row: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        t0 = self.attempt["t0"]
        with open(self.folder / row["file"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(self.COLUMNS)
            for i in range(len(r.t)):
                tx, ty = r.actual_tilt[i]
                alphas = r.alphas[i] or (math.nan,) * 3
                torques = r.torques[i] if r.torques and r.torques[i] else (math.nan,) * 3
                values = [r.t[i] - t0, 1000 * r.x[i], 1000 * r.y[i], 1000 * r.height[i], *r.velocity[i],
                           1000 * r.x_ref[i], 1000 * r.y_ref[i],
                           1000 * math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]),
                           math.degrees(r.theta_x[i]), math.degrees(r.theta_y[i]),
                           math.degrees(tx), math.degrees(ty), *(math.degrees(a) for a in alphas),
                           *torques, 100 * r.airborne[i] if r.airborne else math.nan,
                           1000 * r.measurement_error[i]]
                w.writerow([_format_number(v, 4) for v in values])
        keys = list(row.keys())
        for l in self.summary:                  # columns of all the attempts (throw ⊃ free)
            keys += [k for k in l if k not in keys]
        with open(self.folder / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(keys)
            for l in self.summary:
                w.writerow([_format_number(l.get(k), 3) if isinstance(l.get(k), float) else
                            ("yes" if l.get(k) is True else "no" if l.get(k) is False else
                             "" if l.get(k) is None else l.get(k)) for k in keys])

    # --- Simulation step ---------------------------------------------------------------------
    def step(self) -> None:
        """Advances by one step of dt_physics, whatever happens to the ball."""
        sim, phys = self.sim, self.phys
        if self.state == "fallen":
            # No more ball to track: plate brought back level, nothing is recorded
            if sim.n % self.ppc == 0:
                phys.command(0.0, 0.0)
                phys.read_period_airborne_steps()     # not recorded
            phys.step(self.dt)
            sim.n += 1
            return
        control = sim.n % self.ppc == 0
        if not sim.step():                       # ball fallen (detected before the step)
            if self.attempt is not None:
                self._finish("fallen")
            else:
                self.state = "fallen"
                self.message = "The ball has fallen."
            return
        h = phys.ball_height()
        if self.attempt is not None:
            self._track_step(h)
        if control:
            self._after_control(h)

    def _track_step(self, h: float) -> None:
        """Landings (after a flight of more than MIN_FLIGHT above RESTING_HEIGHT) and maximum height of
        the bounces, checked at every step: sampling at 50 Hz would miss them."""
        att = self.attempt
        if h > RESTING_HEIGHT:
            self._steps_in_flight += 1
            if att["landings"]:
                att["max_bounce"] = max(att["max_bounce"], h)
        else:
            if self._steps_in_flight * self.dt >= MIN_FLIGHT:
                att["landings"] += 1
            self._steps_in_flight = 0

    def _after_control(self, h: float) -> None:
        r = self.sim.r
        d = math.hypot(r.x[-1] - r.x_ref[-1], r.y[-1] - r.y_ref[-1])
        self.curves.append((r.t[-1], d, r.theta_x[-1], r.theta_y[-1], h, r.x[-1], r.y[-1],
                             r.x_ref[-1], r.y_ref[-1]))
        if self.attempt is None:
            # Waiting: keep 0.5 s of history, and a disturbance starts an attempt
            n_before = round(PRE_DISTURBANCE / self.cfg.control_period)
            if len(r.t) > 20 * n_before:
                self.sim.r = _keep_last(r, n_before)
            if d > DISTURBANCE_THRESHOLD:
                self._new_attempt("disturbance", {}, before=_keep_last(self.sim.r, n_before))
                self.attempt["t0"] = self.sim.r.t[-1]
            return
        att = self.attempt
        if d < STABLE_THRESHOLD and h < RESTING_HEIGHT:
            if att["stable_since"] is None:
                att["stable_since"] = r.t[-1]
            elif r.t[-1] - att["stable_since"] >= STABLE_DURATION:
                self._finish("stabilized")
                return
        else:
            att["stable_since"] = None
        if self.t - att["t0"] >= self.max_duration:
            self._finish("not stabilized")

    def close(self) -> list:
        """Ends the current attempt; returns the summary of all the attempts."""
        if self.attempt is not None:
            self._finish("interrupted")
        self.close_camera()
        return self.summary


class RealTime:
    """Keeps a live session in step with real time. Called once per displayed frame, with the real time `now` (s):
    the simulation catches up with the real time elapsed since the previous frame, up to max_lag; beyond, it slows
    down instead of catching up (slow computer, window being moved...). The simulation advances by whole steps: what
    a frame advances beyond the real time is taken off the next one, otherwise the simulation would run ahead of
    real time by half a step per frame on average (1.5 % with steps of 0.5 ms at 60 frames per second). Measures,
    over about 1 s, the speed (simulated time / real time) and the number of frames per second."""

    MEASURE = 1.0       # (s) of real time per measurement of the speed and of the frames per second

    def __init__(self, max_lag: float = 0.05):
        self.max_lag = max_lag
        self.paused = False
        self.speed = 1.0                   # simulated time / real time, measured
        self.frames_per_second = 0.0       # measured
        self._clock = None                 # real time of the previous frame
        self._ahead = 0.0                  # (s) simulated beyond the real time by the previous frame (< one step)
        self._mark = None                  # (real time, simulated time, frames) at the start of the measurement
        self._frames = 0

    def advance(self, session, now: float) -> None:
        """Advances `session` (an object with t and step()) by the real time elapsed since the previous call.
        The first call only starts the clock."""
        if self._clock is None:
            self._clock, self._mark = now, (now, session.t, 0)
            return
        elapsed, self._clock = min(now - self._clock, self.max_lag), now
        self._frames += 1
        if self.paused:
            self._ahead = 0.0
            self._mark = (now, session.t, self._frames)
            return
        target = session.t + elapsed - self._ahead
        while session.t < target:
            session.step()
        self._ahead = session.t - target
        real0, sim0, frames0 = self._mark
        if now - real0 > self.MEASURE:
            self.speed = (session.t - sim0) / (now - real0)
            self.frames_per_second = (self._frames - frames0) / (now - real0)
            self._mark = (now, session.t, self._frames)
