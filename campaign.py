"""Measurement campaign: consistency of the models, robustness, latency. Writes results/report.md.

Reference chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.
The 160×120 camera is as accurate as 320×240, and faster to simulate.

    0. From the simple model to the full chain: what each level of realism adds
    1. Theoretical limits: no-slip condition; bounce compliant with the ITTF rule
    2. Thrown ball: maximum speed caught (max tilt 10°, then 20°)
    3. Push on a ball at rest at the center: maximum impulse absorbed
    4. Robustness to physical parameters and to a dropped ball
    5. Dropped ball: where and from how high
    6. Camera latency, with and without compensation by prediction
    7. Integral action: error integrated everywhere or only near the setpoint
    8. Gains of the PID: hand-tuned or by LQR
    9. Square trajectory
    10. Gap between the simple physics and MuJoCo, explained term by term (circle and return), on the
        angle-driven plate and on the robot
    11. Adaptive control: the model gain c·g estimated online, for a hollow and a solid ball
    12. The remaining gaps of more than 0.5 mm, reproduced by our equations or traced to their cause
    13. The AI brain (learned policy) next to the PID brain: same objectives, same cases, same draws

Every case is simulated over N draws, so that each result comes with its spread:
    - draw 0 is the nominal configuration (noise seed 0, nominal start position);
    - draw k ≥ 1 uses noise seed k and a start position shifted by a uniform offset of up to
      ±0.5 mm on each axis. The camera chain draws no random number: for the camera cases, the
      start offset is the only source of variation.
All simulations are independent: they run in parallel, one process per core by default.

Run (from ball-plate-sim, environment activated):
    python campaign.py                  10 draws (a few minutes)
    python campaign.py --draws 1        nominal configuration only (one simulation per case)
    python campaign.py --processes 4    limit the number of processes
"""
import argparse
from dataclasses import replace
import math
import os
import random
import statistics
import time

import matplotlib.pyplot as plt

from simulation.config import G, REFERENCE_CONTROLLER, REFERENCE_GAINS, Config, CameraParams, square, center
from simulation.loop import simulate
from simulation.metrics import (arrival_time, distance, max_distance, overshoot, stoppable_speed, theta_std,
                                tracking_error_rms)
from simulation.mujoco_physics import rolling_acceleration, measured_restitution
from simulation.parallel import execute

BASE = Config(sensor="camera", camera=CameraParams(width=160, image_height=120))
PM = BASE.mujoco_params
K = BASE.plate.k_inertia
CG = BASE.plate.coefficient * G
R_PLATE = BASE.plate.plate_radius
THROW_START = -0.10                 # the ball is thrown from x = -10 cm toward +x
DROP_HEIGHT = 0.305                 # (m) ITTF rule: height of the bounce test
DROP_DISTANCES = (0.0, 0.05, 0.10)  # (m) section 5: from the center, toward the start point of the reference scenario
DROP_HEIGHTS = (0.10, 0.20, DROP_HEIGHT)
DROP_DURATION = 6.0                 # (s)
AIRBORNE = 0.015                    # (m) section 5: the ball counts as in the air above this height
START_OFFSET = 0.0005               # (m) draws k ≥ 1: start position shifted by up to ±0.5 mm per axis
SPEED_STEP = 0.01                   # (m/s) resolution of the maximum speeds
SPEED_RANGE = (0.1, 1.5)            # (m/s) bisection bracket: caught at the low end, fallen at the high end
CHECK_SPAN = 5                      # draw 0: every speed within ±5 steps of the threshold is simulated


def with_calibration_error(cfg: Config, factor: float) -> Config:
    """The same configuration, with the calibration error of the plate multiplied by factor."""
    p = cfg.plate
    return replace(cfg, plate=replace(p, bias_x=factor * p.bias_x, bias_y=factor * p.bias_y))


HAND_GAINS = replace(BASE, gains=REFERENCE_GAINS)    # the brain with the hand-tuned gains used before B3

# Section 12: our equations (simple physics, 3-arm robot, full equation of section 10 with sliding) and MuJoCo,
# both with the brain and a sensor without noise
FULL_EQUATION = {"rolling_term": True, "frame_terms": True, "exact_gravity": True, "servo_lag": True}
NOISE_FREE = {"sigma": 0.0, "sigma_height": 0.0}
EQUATIONS = Config(physics="simple", sliding=True, **FULL_EQUATION, **NOISE_FREE)
MUJOCO_NOISE_FREE = Config(**NOISE_FREE)
CAMERA_SIGMA = 0.0002               # (m) synthetic sensor with the mean error of the camera (0.25 mm = σ·√(π/2))
VARIANTS = {"solid_ball": {"k_actual": 2 / 5}, "no_calibration_error": None, "calibration_error_x2": None,
            "friction_0.02": {"friction": 0.02}}


def variant(cfg: Config, name: str) -> Config:
    """cfg with the physical change of the robustness variant `name` (section 4)."""
    if name == "no_calibration_error":
        return with_calibration_error(cfg, 0.0)
    if name == "calibration_error_x2":
        return with_calibration_error(cfg, 2.0)
    return replace(cfg, mujoco_params=replace(cfg.mujoco_params, **VARIANTS[name]))


def throw_config(cfg: Config, speed: float, tilt_deg: float = 10.0) -> Config:
    """Ball thrown from x = THROW_START toward +x at `speed`, setpoint at the center."""
    return replace(cfg, theta_max=math.radians(tilt_deg), trajectory=center, duration=5.0,
                   start_position=(THROW_START, 0.0), start_velocity=(speed, 0.0))


def drop_case(distance: float, height: float, aimed: bool = True) -> str:
    return f"drop_{round(1000 * distance)}mm_{round(1000 * height)}mm" + ("" if aimed else "_reacting")


def drop_config(distance: float, height: float) -> Config:
    """Ball released without speed `height` above the plate, at `distance` from the center toward the start
    point of the reference scenario; setpoint at the center."""
    x0, y0 = BASE.start_position
    d0 = math.hypot(x0, y0)
    return replace(BASE, trajectory=center, duration=DROP_DURATION, start_height=height,
                   start_position=(x0 * distance / d0, y0 * distance / d0))


# Closed-loop cases (nominal configurations), each simulated over every draw
CASES = {
    # Simulator validation (sections 0 and 10), frozen reference controller (simulation/config.py)
    "simple_plate": Config(physics="simple", mechanism=False, **REFERENCE_CONTROLLER),
    "simple_plate_rolling": Config(physics="simple", mechanism=False, rolling_term=True, **REFERENCE_CONTROLLER),
    "simple_plate_frame": Config(physics="simple", mechanism=False, frame_terms=True, **REFERENCE_CONTROLLER),
    "simple_plate_gravity": Config(physics="simple", mechanism=False, exact_gravity=True, **REFERENCE_CONTROLLER),
    "simple_plate_servo": Config(physics="simple", mechanism=False, servo_lag=True, **REFERENCE_CONTROLLER),
    "simple_plate_full": Config(physics="simple", mechanism=False, rolling_term=True, frame_terms=True,
                                exact_gravity=True, servo_lag=True, **REFERENCE_CONTROLLER),
    "simple_robot": Config(physics="simple", **REFERENCE_CONTROLLER),
    "simple_robot_rolling": Config(physics="simple", rolling_term=True, **REFERENCE_CONTROLLER),
    "simple_robot_frame": Config(physics="simple", frame_terms=True, **REFERENCE_CONTROLLER),
    "simple_robot_gravity": Config(physics="simple", exact_gravity=True, **REFERENCE_CONTROLLER),
    "simple_robot_servo": Config(physics="simple", servo_lag=True, **REFERENCE_CONTROLLER),
    "simple_robot_full": Config(physics="simple", rolling_term=True, frame_terms=True, exact_gravity=True,
                                servo_lag=True, **REFERENCE_CONTROLLER),
    "simple_robot_full_noise_free": Config(physics="simple", rolling_term=True, frame_terms=True, exact_gravity=True,
                                           servo_lag=True, sigma=0.0, sigma_height=0.0, **REFERENCE_CONTROLLER),
    "mujoco_plate": Config(mechanism=False, **REFERENCE_CONTROLLER),
    "mujoco_robot": Config(**REFERENCE_CONTROLLER),
    "mujoco_robot_noise_free": Config(sigma=0.0, sigma_height=0.0, **REFERENCE_CONTROLLER),
    "camera_reference": replace(BASE, **REFERENCE_CONTROLLER),
    # The brain (default settings), full chain with the camera
    "camera": BASE,
    "solid_ball": replace(BASE, mujoco_params=replace(PM, k_actual=2 / 5)),
    "heavy_ball": replace(BASE, mujoco_params=replace(PM, ball_mass=0.027)),
    "friction_0.2": replace(BASE, mujoco_params=replace(PM, friction=0.2)),
    "friction_0.02": replace(BASE, mujoco_params=replace(PM, friction=0.02)),
    "heavy_plate": replace(BASE, mujoco_params=replace(PM, plate_mass=0.5)),
    "dropped": replace(BASE, start_height=DROP_HEIGHT),
    "no_calibration_error": with_calibration_error(BASE, 0.0),
    "calibration_error_x2": with_calibration_error(BASE, 2.0),
    "latency_60ms": replace(BASE, latency=3),
    "latency_100ms": replace(BASE, latency=5),
    "square": replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05)),
    # Dropped ball (section 5): the brain, and the brain reacting in the air as on the plate (before B5)
    **{drop_case(d, h): drop_config(d, h) for d in DROP_DISTANCES for h in DROP_HEIGHTS},
    **{drop_case(d, h, aimed=False): replace(drop_config(d, h), airborne_height=math.inf)
       for d in DROP_DISTANCES for h in DROP_HEIGHTS},
    # The brain without one of its parts: latency compensation (sections 6 and 9), fast ramps kept for a fast
    # ball (section 6), integration zone (section 7), LQR gains (section 8)
    "camera_uncompensated": replace(BASE, latency_compensation=False),
    "latency_60ms_uncompensated": replace(BASE, latency=3, latency_compensation=False),
    "latency_100ms_uncompensated": replace(BASE, latency=5, latency_compensation=False),
    "square_uncompensated": replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05),
                                    latency_compensation=False),
    "camera_integral_everywhere": replace(BASE, integration_zone=math.inf),
    "camera_fast_always": replace(BASE, fast_speed=0.0),
    "latency_100ms_fast_always": replace(BASE, latency=5, fast_speed=0.0),
    "camera_hand_gains": HAND_GAINS,
    "no_calibration_error_hand_gains": with_calibration_error(HAND_GAINS, 0.0),
    "calibration_error_x2_hand_gains": with_calibration_error(HAND_GAINS, 2.0),
    "latency_60ms_hand_gains": replace(HAND_GAINS, latency=3),
    # The brain with the model gain fixed (section 11)
    "camera_fixed_gain": replace(BASE, adapt_model_gain=False),
    "latency_60ms_fixed_gain": replace(BASE, latency=3, adapt_model_gain=False),
    "solid_ball_fixed_gain": replace(BASE, mujoco_params=replace(PM, k_actual=2 / 5), adapt_model_gain=False),
    # Section 12: the robustness variants without noise, in MuJoCo (also with its pyramidal friction cone) and in
    # our equations
    "mujoco_noise_free": MUJOCO_NOISE_FREE,
    **{f"mujoco_noise_free_{name}": variant(MUJOCO_NOISE_FREE, name) for name in VARIANTS},
    "mujoco_noise_free_pyramidal": replace(MUJOCO_NOISE_FREE, mujoco_params=replace(PM, cone="pyramidal")),
    "mujoco_noise_free_pyramidal_friction_0.02": replace(MUJOCO_NOISE_FREE,
                                                         mujoco_params=replace(PM, cone="pyramidal", friction=0.02)),
    "equations": EQUATIONS,
    **{f"equations_{name}": variant(EQUATIONS, name) for name in VARIANTS},
    # ... the effect of the mechanism (section 0), without noise: the arms halve the ramps of the plate
    "mechanism_plate": Config(physics="simple", mechanism=False, **NOISE_FREE, **REFERENCE_CONTROLLER),
    "mechanism_robot": Config(physics="simple", **NOISE_FREE, **REFERENCE_CONTROLLER),
    "mechanism_robot_double_ramps": Config(physics="simple", max_acceleration=2 * BASE.max_acceleration,
                                           max_speed=2 * BASE.max_speed, **NOISE_FREE, **REFERENCE_CONTROLLER),
    # ... 100 ms of latency without compensation (section 6): our equations, full and basic
    "latency_100ms_uncompensated_equations": replace(EQUATIONS, latency=5, latency_compensation=False,
                                                     sigma=CAMERA_SIGMA),
    "latency_100ms_uncompensated_basic": Config(physics="simple", latency=5, latency_compensation=False,
                                                sigma=CAMERA_SIGMA),
    # ... the thrown ball with and without waiting for its speed (section 2), at the same speed
    "throw_0.55_wait": throw_config(BASE, 0.55),
    "throw_0.55_no_wait": throw_config(replace(BASE, wait_for_speed=False), 0.55),
}
# The validation of the simulator (sections 0 and 10, and the effect of the noise in section 12) is simulated over
# The AI brain (section 13): the learned policy brains/<AI>.npz, on the cases of the objectives of the PID brain
AI = "imitation"
AI_CASES = ("camera", "solid_ball", "heavy_ball", "friction_0.2", "friction_0.02", "heavy_plate", "dropped",
            "no_calibration_error", "calibration_error_x2", "latency_60ms", "latency_100ms")
CASES.update({f"{name}_ai": replace(CASES[name], brain=AI) for name in AI_CASES})
# 4 times more draws: with the reference controller, the noise of the synthetic sensor makes the return vary by
# more than 1 mm from one draw to the next, and a share of 80 % would be uncertain by ±15 % over 10 draws
DRAWS_FACTOR = {name: 4 for name in CASES if name.startswith(("simple_plate", "simple_robot"))
                or name in ("mujoco_plate", "mujoco_robot", "mujoco_robot_noise_free", "camera_reference")}
ROBUSTNESS = {"Nominal (ping-pong, μ = 1)": "camera", "Solid ball (k = 2/5)": "solid_ball",
              "Ball 10× heavier (27 g)": "heavy_ball", "Friction μ = 0.2": "friction_0.2",
              "Friction μ = 0.02": "friction_0.02", "Plate 2× heavier (0.5 kg)": "heavy_plate",
              "No calibration error (0°, 0°)": "no_calibration_error",
              "Calibration error ×2 (1°, -0.6°)": "calibration_error_x2",
              "Ball dropped from 30.5 cm (ITTF height)": "dropped"}
# The return to the center must hold whatever the calibration error of the plate, from 0 to 1°
CALIBRATION = {"none": "no_calibration_error", "nominal": "camera", "×2": "calibration_error_x2"}
CALIBRATION_HAND_GAINS = {"none": "no_calibration_error_hand_gains", "nominal": "camera_hand_gains",
                          "×2": "calibration_error_x2_hand_gains"}
# Maximum speeds searched by bisection: name -> (kind, max tilt in degrees, settings changed from BASE)
LIMITS = {"throw_10deg": ("throw", 10, {}), "throw_20deg": ("throw", 20, {}), "push": ("push", 10, {}),
          "throw_10deg_no_wait": ("throw", 10, {"wait_for_speed": False}),
          "throw_10deg_slow_motors": ("throw", 10, {"fast_move": math.inf}),
          "throw_10deg_hand_gains": ("throw", 10, {"gains": REFERENCE_GAINS}),
          "throw_10deg_ai": ("throw", 10, {"brain": AI}), "push_ai": ("push", 10, {"brain": AI})}


# --- Draws and statistics ------------------------------------------------------------------------
def draw(cfg: Config, k: int) -> Config:
    """Configuration of draw k. Draw 0: cfg unchanged (nominal). Draw k ≥ 1: noise seed k, and start
    position shifted by a uniform offset in ±START_OFFSET on each axis. The offset has its own random
    generator, seeded with a string (reproducible on every machine) and independent of the noise."""
    if k == 0:
        return cfg
    rng = random.Random(f"start offset {k}")
    x, y = cfg.start_position
    return replace(cfg, seed=k, start_position=(x + rng.uniform(-START_OFFSET, START_OFFSET),
                                                y + rng.uniform(-START_OFFSET, START_OFFSET)))


def mean_std(values):
    """Mean and sample standard deviation, ignoring NaN (draws where the ball fell).
    The standard deviation is NaN with fewer than two values."""
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return math.nan, math.nan
    return statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else math.nan


# 97.5 % quantiles of Student's t distribution, for 1 to 39 degrees of freedom (1.96 beyond)
T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
         2.040, 2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023)


def paired_delta(a, b):
    """Difference b - a over the same draws: (mean, standard error, significant).

    Paired t-test at the 5 % level: significant when |mean| > t × standard error, t being the
    97.5 % quantile of Student's distribution with n - 1 degrees of freedom (2.26 for n = 10).
    Needs at least two draws."""
    d = [y - x for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(d) < 2:
        return (d[0] if d else math.nan), math.nan, False
    m, se = statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))
    t = T_975[len(d) - 2] if len(d) - 1 <= len(T_975) else 1.96
    return m, se, abs(m) > t * se


def max_caught(caught, low: float, high: float, step: float):
    """Largest multiple of step in [low, high] for which caught(v) is true, by bisection.

    Assumes the ball is caught below a threshold and falls above it (checked separately on draw 0).
    Returns (v, tested), tested being the list of (v, caught(v)) in the order of the tests.
    v is None if caught(low) is false, and high if caught(high) is true (threshold out of range).
    """
    lo, hi = round(low / step), round(high / step)        # integer multiples of step: no drift
    tested = []

    def test(i: int) -> bool:
        v = round(i * step, 10)
        ok = caught(v)
        tested.append((v, ok))
        return ok

    if not test(lo):
        return None, tested
    if test(hi):
        return round(hi * step, 10), tested
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if test(mid):
            lo = mid
        else:
            hi = mid
    return round(lo * step, 10), tested


def limit_config(name: str, v: float) -> Config:
    """Thrown ball at speed v, or push of v at t = 1 s on the ball at rest at the center."""
    kind, tilt_deg, settings = LIMITS[name]
    cfg = replace(BASE, **settings)
    if kind == "throw":
        return throw_config(cfg, v, tilt_deg)
    return replace(cfg, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                   pushes=((1.0, 0.8 * v, 0.6 * v),))


def physical_limit(name: str) -> float:
    """Highest speed that the plate can stop over the available distance d."""
    kind, tilt_deg, _ = LIMITS[name]
    distance = R_PLATE - THROW_START if kind == "throw" else R_PLATE
    return stoppable_speed(distance, math.radians(tilt_deg), CG)


def find_limit(job):
    """Maximum speed caught for one draw. job = (limit name, draw k).
    Returns (v_max, tested, runs): runs keeps the full histories of draw 0 (for the figures)."""
    name, k = job
    runs = {}

    def caught(v: float) -> bool:
        r = simulate(draw(limit_config(name, v), k))
        if k == 0:
            runs[v] = r
        return not r.fallen

    v_max, tested = max_caught(caught, *SPEED_RANGE, SPEED_STEP)
    return v_max, tested, runs


# --- Indicators (NaN when the ball fell) ---------------------------------------------------------
def circle_error(r) -> float:
    return math.nan if r.fallen else tracking_error_rms(r, 12, 20)


def square_error(r) -> float:
    return math.nan if r.fallen else tracking_error_rms(r, 8, 16)


def return_distance(r) -> float:
    """Objective "back to the center in less than 2 s": max distance between 2 s and 4 s ≤ 5 mm
    (reference scenario: start 10 cm away, setpoint at the center until 4 s)."""
    return math.nan if r.fallen else max_distance(r, 2, 4)


def arrival(r) -> float:
    return math.nan if r.fallen else arrival_time(r, 4)


def overshoot_after_arrival(r) -> float:
    return math.nan if r.fallen else overshoot(r, 4)


def final_model_gain(r) -> float:
    """Model gain c·g used by the controller at the end of the scenario."""
    return math.nan if r.fallen else r.model_gain[-1]


def jitter(r) -> float:
    return math.nan if r.fallen else theta_std(r, 2.5, 4)


def airborne(r) -> float:
    return math.nan if r.fallen else r.airborne_fraction


def height_error(r) -> float:
    """RMS error of the measured height while the ball is in the air (true height above AIRBORNE)."""
    if r.fallen:
        return math.nan
    errors = [m - h for m, h in zip(r.measured_height, r.height) if h > AIRBORNE]
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan


def false_airborne(r) -> float:
    """Share of the images where the ball is on the plate (true height < 5 mm) but measured in the air."""
    if r.fallen:
        return math.nan
    on_plate = [m for m, h in zip(r.measured_height, r.height) if h < 0.005]
    return sum(m > AIRBORNE for m in on_plate) / len(on_plate) if on_plate else math.nan


def settled_distance(r) -> float:
    """Dropped ball: max distance from the setpoint over the last 2 s of the drop."""
    return math.nan if r.fallen else max_distance(r, DROP_DURATION - 2, DROP_DURATION)


def first_pass_minimum(r) -> float:
    """Closest approach to the setpoint before the first overshoot: minimum distance from the first time the ball
    comes within 1 cm until it is more than 1 cm away again (NaN if it never comes within 1 cm)."""
    d = [distance(r, i) for i in range(len(r.t))]
    start = next((i for i, v in enumerate(d) if v < 0.01), None)
    if r.fallen or start is None:
        return math.nan
    end = next((i for i in range(start, len(d)) if d[i] > 0.01), len(d))
    return min(d[start:end])


def last_flight(r) -> float:
    """Last time the ball is in the air (true height above AIRBORNE), NaN if never."""
    return next((t for t, h in zip(reversed(r.t), reversed(r.height)) if h > AIRBORNE), math.nan)


def braking(r, tilt: float = math.radians(-9.5)):
    """Thrown ball (toward +x): first time the plate tilts beyond `tilt` against it, and where the ball is then."""
    i = next((i for i, (tx, _) in enumerate(r.actual_tilt) if tx <= tilt), None)
    return (math.nan, math.nan) if i is None else (r.t[i], r.x[i])


# --- Formatting ----------------------------------------------------------------------------------
def fmt(values, unit: str = "", decimals: int = 2, scale: float = 1.0) -> str:
    """'mean ± std unit' over the draws ('mean unit' with a single draw)."""
    m, s = mean_std([v * scale for v in values])
    if math.isnan(m):
        return "-"
    text = f"{m:.{decimals}f}" if math.isnan(s) else f"{m:.{decimals}f} ± {s:.{decimals}f}"
    return text + unit


def fmt_delta(a, b, unit: str = "", decimals: int = 2, scale: float = 1.0) -> str:
    """Paired difference b - a: 'mean ± standard error unit', in bold when significant."""
    m, se, significant = paired_delta([v * scale for v in a], [v * scale for v in b])
    if math.isnan(m):
        return "-"
    text = f"{m:+.{decimals}f}" if math.isnan(se) else f"{m:+.{decimals}f} ± {se:.{decimals}f}"
    return f"**{text}{unit}**" if significant else text + unit


def fallen(runs) -> str:
    return f"{sum(r.fallen for r in runs)}/{len(runs)}"


class Report:
    """Markdown report, also printed to the console."""

    def __init__(self, title: str):
        self.md = [f"# {title}\n"]

    def text(self, line: str) -> None:
        print(line)
        self.md.append(f"{line}\n")

    def section(self, title: str) -> None:
        print(f"\n{title}")
        self.md.append(f"\n## {title}\n")

    def table(self, headers, rows) -> None:
        widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
        for row in [headers] + rows:
            print("  " + "  ".join(str(x).rjust(w) if i else str(x).ljust(w)
                                   for i, (x, w) in enumerate(zip(row, widths))))
        self.md.append("| " + " | ".join(headers) + " |")
        self.md.append("|" + "---|" * len(headers))
        self.md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
        self.md.append("")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.md) + "\n")


# --- Computation -----------------------------------------------------------------------------------
def run_campaign(draws: int, processes: int = None) -> dict:
    """Runs every simulation of the campaign. Returns:
        runs[case]            list of Results, one per draw
        limits[name]          list of (v_max, tested, runs of draw 0), one per draw
        checks[name]          draw 0: list of (v, caught) around the threshold
    """
    jobs = [(name, k) for name in CASES for k in range(draws * DRAWS_FACTOR.get(name, 1))]
    results = execute(simulate, [draw(CASES[name], k) for name, k in jobs], processes)
    runs = {name: [] for name in CASES}
    for (name, _), r in zip(jobs, results):
        runs[name].append(r)

    jobs = [(name, k) for name in LIMITS for k in range(draws)]
    limits = {name: [] for name in LIMITS}
    for (name, _), result in zip(jobs, execute(find_limit, jobs, processes)):
        limits[name].append(result)

    # Monotonicity check on draw 0: every speed within ±CHECK_SPAN steps of the threshold
    jobs = []
    for name in LIMITS:
        v_max = limits[name][0][0]
        if v_max is not None:
            jobs += [(name, round(v_max + i * SPEED_STEP, 10)) for i in range(-CHECK_SPAN, CHECK_SPAN + 1)]
    checks = {name: [] for name in LIMITS}
    for (name, v), r in zip(jobs, execute(simulate, [limit_config(name, v) for name, v in jobs], processes)):
        checks[name].append((v, not r.fallen))
    return {"draws": draws, "runs": runs, "limits": limits, "checks": checks}


def worst_return(runs: dict, calibration: dict, suffix: str = "") -> tuple:
    """(label, return distances over the draws) of the calibration error whose mean return is the worst
    (a calibration error where every draw fell counts as the worst). suffix: see objectives."""
    backs = {label: [return_distance(r) for r in runs[case + suffix]] for label, case in calibration.items()}
    means = {label: mean_std(values)[0] for label, values in backs.items()}
    worst = max(backs, key=lambda label: math.inf if math.isnan(means[label]) else means[label])
    return worst, backs[worst]


def objectives(data: dict, suffix: str = "") -> list:
    """Quantified objectives of the brain, judged on the mean over the draws: rows [objective, measured,
    target, met]. Thrown ball at ±10°, circle at 60 ms of latency, return to the center (worst of the
    calibration errors), robustness. suffix: "" for the PID brain, "_ai" for the AI brain (same cases)."""
    runs = data["runs"]
    shares = [math.nan if v is None else v / physical_limit("throw_10deg")
              for v, _, _ in data["limits"]["throw_10deg" + suffix]]
    circle = [circle_error(r) for r in runs["latency_60ms" + suffix]]
    worst, back = worst_return(runs, CALIBRATION, suffix)
    falls = sum(r.fallen for case in ROBUSTNESS.values() for r in runs[case + suffix])
    total = sum(len(runs[case + suffix]) for case in ROBUSTNESS.values())
    met = lambda ok: "yes" if ok else "**no**"
    return [
        ["Thrown ball caught (±10°)", fmt(shares, " %", 0, 100), "≥ 90 % of the physical limit",
         met(mean_std(shares)[0] >= 0.90)],
        ["Circle with 60 ms of latency", fmt(circle, " mm", scale=1000), "< 2 mm", met(mean_std(circle)[0] < 0.002)],
        ["Back to the center in less than 2 s", f"{fmt(back, ' mm', scale=1000)} (calibration error: {worst})",
         "≤ 5 mm between 2 s and 4 s", met(mean_std(back)[0] <= 0.005)],
        ["No fall in the robustness variants", f"{falls}/{total} falls", "0", met(falls == 0)],
    ]


def monotonic(name: str, data: dict) -> str:
    """Draw 0: are the speeds around the threshold caught below it and fallen above it?"""
    v_max = data["limits"][name][0][0]
    if v_max is None:
        return "not checked (no speed caught)"
    wrong = [v for v, ok in data["checks"][name] if ok != (v <= v_max)]
    return "yes" if not wrong else "NO, exceptions at " + ", ".join(f"{v:.2f}" for v in wrong) + " m/s"


# --- Report --------------------------------------------------------------------------------------------
def write_report(data: dict, path: str = os.path.join("results", "report.md")) -> None:
    n, runs = data["draws"], data["runs"]
    rep = Report("Measurement campaign report for the simulated balancing robot")
    rep.text("Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.")
    rep.text(f"Each case is simulated over {n} draw(s). Draw 0 is the nominal configuration; draw k ≥ 1 "
             "uses noise seed k and a start position shifted by up to ±0.5 mm on each axis "
             "(the only source of variation for the camera cases). Values: mean ± standard deviation "
             "over the draws; draws where the ball fell are counted in 'Fallen' and left out of the "
             "other columns. Δ: mean of the paired differences over the same draws ± standard error, "
             "in **bold** when significant (paired t-test at the 5 % level: |Δ| > t × standard error, "
             "t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws). Sections 0 and 10, which validate "
             f"the simulator: {DRAWS_FACTOR['mujoco_robot'] * n} draws.")
    kp, ki, kd = BASE.pid_gains()
    rep.text(f"The brain is the PID with its LQR gains (Kp = {kp:.2f}, Ki = {ki:.2f}, Kd = {kd:.2f}), its latency "
             "compensation, its integration zone and its aimed bounce (default settings): it is judged by the "
             "objectives and sections 2 to 9. Sections 0 and 10 validate the simulator: they use a frozen reference "
             "controller (hand-tuned gains Kp = 1, Ki = 1, ζ = 0.7, without latency compensation, error integrated "
             "everywhere, ball never considered in the air) and the motors of the first robot (single ramp).")

    rep.section("Objectives of the brain (full chain with the camera)")
    rep.table(["Objective", "Measured", "Target", "Met"], objectives(data))
    rep.text("Judged on the mean over the draws. Thrown ball: maximum speed caught at ±10° divided by the "
             "physical limit (section 2). Return: start 10 cm away, setpoint at the center, worst of the three "
             "calibration errors of the plate of section 4 (none, nominal 0.5°/-0.3°, ×2), since the actual one "
             "is unknown. Robustness: every variant and every draw of section 4.")

    rep.section("0. From the simple model to the full chain (reference scenario, reference controller)")
    # Each level is compared with the level that differs from it by a single factor
    levels = (("Simple physics: angle-driven plate", "simple_plate", None, "-"),
              ("Simple physics: 3-arm robot", "simple_robot", "simple_plate", "mechanism"),
              ("MuJoCo: angle-driven plate", "mujoco_plate", "simple_plate", "physics"),
              ("MuJoCo: 3-arm robot", "mujoco_robot", "simple_robot", "physics"),
              ("MuJoCo: 3-arm robot, sensor without noise", "mujoco_robot_noise_free", "mujoco_robot", "noise"),
              ("MuJoCo: 3-arm robot + camera", "camera_reference", "mujoco_robot_noise_free", "camera"))
    rows = []
    for label, case, compared, factor in levels:
        rs = runs[case]
        delta = "-" if compared is None else fmt_delta([return_distance(r) for r in runs[compared]],
                                                       [return_distance(r) for r in rs], " mm", scale=1000)
        rows.append([label, fallen(rs), fmt([arrival(r) for r in rs], " s"),
                     fmt([overshoot_after_arrival(r) for r in rs], " mm", scale=1000),
                     fmt([return_distance(r) for r in rs], " mm", scale=1000), factor, delta])
    rep.table(["Level", "Fallen", "Arrival (first time < 5 mm)", "Overshoot after arrival",
               "Max distance between 2 s and 4 s", "Factor changed", "Δ (effect of the factor)"], rows)
    rep.text("Return to the center (start 10 cm away, setpoint at the center until 4 s). Objective "
             "\"back to the center in less than 2 s\": max distance between 2 s and 4 s ≤ 5 mm. Angle-driven plate: "
             "its two angles follow the command with the profile of the motors (speed, ramps), without steps; "
             "the mechanism adds the arms (kinematics, microsteps: the motors turn about twice as much as the "
             "plate, so their ramps tilt it more slowly).")
    rows = []
    for label, case, compared, factor in levels:
        rs = runs[case]
        delta = "-" if compared is None else fmt_delta([circle_error(r) for r in runs[compared]],
                                                       [circle_error(r) for r in rs], " mm", scale=1000)
        contact = fmt([airborne(r) for r in rs], " %", 1, 100) if CASES[case].physics == "mujoco" else "-"
        rows.append([label, fmt([circle_error(r) for r in rs], " mm", scale=1000), factor, delta,
                     fmt([jitter(r) for r in rs], "°", scale=180 / math.pi), contact])
    rep.table(["Level", "Circle error", "Factor changed", "Δ circle error (effect of the factor)",
               "θx jitter", "Airborne"], rows)
    rep.text("Circle: 6 cm radius in 4 s, error = root-mean-square distance from 12 s to 20 s. "
             "Factor changed: compared with the level above that differs only by this factor "
             "(the mechanism: the simple plate; the physics: the same level in simple physics; the noise: the "
             "synthetic sensor of standard deviation 1 mm, made perfect; the camera: the perfect sensor). The "
             "synthetic sensor is about 5 times noisier than the camera: its noise, not the camera, makes the gap "
             "between them.")

    rep.section("1. Theoretical limits and contact calibration")
    mu_10 = K * math.tan(math.radians(10)) / (1 + K)
    a_roll = rolling_acceleration(K, math.radians(10), friction=1.3 * mu_10)
    a_slide = rolling_acceleration(K, math.radians(10), friction=0.7 * mu_10)
    rep.table(["Quantity", "Theory", "MuJoCo"], [
        ["Minimum friction to roll at 10°", f"{mu_10:.3f}", "-"],
        ["Acceleration at 10°, friction 1.3×min (rolls)",
         f"{G * math.sin(math.radians(10)) / (1 + K):.3f} m/s²", f"{a_roll:.3f} m/s²"],
        ["Acceleration at 10°, friction 0.7×min (slides)",
         f"{G * (math.sin(math.radians(10)) - 0.7 * mu_10 * math.cos(math.radians(10))):.3f} m/s²",
         f"{a_slide:.3f} m/s²"],
        ["Restitution, drop from 30.5 cm onto a fixed plane (ITTF rule)", "0.85 to 0.91", f"{measured_restitution():.3f}"],
    ])

    def limit_row(name: str, label: str):
        speeds = [math.nan if v is None else v for v, _, _ in data["limits"][name]]
        limit = physical_limit(name)
        return [label, fmt(speeds, " m/s"), f"{limit:.2f} m/s", fmt(speeds, " %", 0, 100 / limit),
                monotonic(name, data)]

    headers = ["Physical limit √(2·c·g·sin θmax·d)", "Share of the limit", "Monotonic around the threshold (draw 0)"]
    rep.section("2. Ball thrown from x = -10 cm toward the opposite edge (setpoint: center)")
    rep.table(["Max tilt", "Max speed caught"] + headers,
              [limit_row("throw_10deg", "±10°"), limit_row("throw_20deg", "±20°"),
               limit_row("throw_10deg_no_wait", "±10°, without waiting for the speed"),
               limit_row("throw_10deg_slow_motors", "±10°, single motor ramp of 50 rad/s² (before B4)")])
    rep.text(f"The first image arrives one latency ({1000 * BASE.latency * BASE.control_period:.0f} ms) after the "
             f"throw; the brain waits for the second one, which gives the speed of the ball, before tilting the "
             f"plate: at the first one, the position alone would tilt it the wrong way. Motor ramps: "
             f"{BASE.fast_acceleration:.0f} rad/s² for the large moves (gap above {BASE.fast_move} rad) of a ball "
             f"faster than {BASE.fast_speed} m/s, {BASE.max_acceleration:.0f} rad/s² otherwise.")
    rep.section("3. Push on the ball at rest at the center (t = 1 s)")
    rep.table(["Max tilt", "Max impulse absorbed (Δv)"] + headers, [limit_row("push", "±10°")])
    rep.text(f"Maximum speeds found by bisection, to {SPEED_STEP:.2f} m/s, between {SPEED_RANGE[0]} and "
             f"{SPEED_RANGE[1]} m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).")

    rep.section("4. Robustness to physical parameters and to a dropped ball (controller unchanged, tuned for "
                "the ping-pong ball)")
    nominal = [circle_error(r) for r in runs["camera"]]
    rows = []
    for label, case in ROBUSTNESS.items():
        rs = runs[case]
        errors = [circle_error(r) for r in rs]
        rows.append([label, fallen(rs), fmt([return_distance(r) for r in rs], " mm", scale=1000),
                     fmt(errors, " mm", scale=1000),
                     "-" if case == "camera" else fmt_delta(nominal, errors, " mm", scale=1000),
                     fmt([airborne(r) for r in rs], " %", 1, 100)])
    rep.table(["Variant", "Fallen", "Max distance between 2 s and 4 s", "Circle error",
               "Δ circle error vs nominal", "Airborne"], rows)
    rep.text(f"Reference scenario. Dropped ball: released without speed above the start point, "
             f"{100 * DROP_HEIGHT:.1f} cm above the plate. A ball in the air beyond the edge of the tilted plate "
             "counts as fallen only once it is level with the plate.")

    rep.section("5. Dropped ball: where and from how high (setpoint: center)")
    rows = []
    for d in DROP_DISTANCES:
        for h in DROP_HEIGHTS:
            rs = runs[drop_case(d, h)]
            rows.append([f"{100 * d:.0f} cm", f"{100 * h:.1f} cm", fallen(runs[drop_case(d, h, aimed=False)]),
                         fallen(rs),
                         fmt([airborne(r) for r in rs], " %", 1, 100),
                         fmt([settled_distance(r) for r in rs], " mm", scale=1000),
                         fmt([height_error(r) for r in rs], " mm", scale=1000),
                         fmt([false_airborne(r) for r in rs], " %", 1, 100)])
    rep.table(["Distance from the center", "Height", "Fallen, reacting in the air", "Fallen, aimed bounce",
               "Airborne",
               f"Max distance from the center between {DROP_DURATION - 2:.0f} s and {DROP_DURATION:.0f} s",
               "Height measured in the air: error (RMS)", "On the plate but measured in the air"], rows)
    rep.text(f"Ball released without speed above a point of the line from the center to the start point of the "
             f"reference scenario, {DROP_DURATION:.0f} s per drop. Reacting in the air: the brain before B5, which "
             f"drives the plate from the measured position of the ball also while it is in the air. Aimed bounce "
             f"(the brain): in the air, the plate is tilted so that the next bounce sends the ball toward the "
             f"setpoint (controller/flight.py, controller/plate.py); the other columns are those of the brain. "
             f"Height: measured by the camera from the apparent size of the "
             f"ball; in the air above {1000 * AIRBORNE:.0f} mm (true height), on the plate below 5 mm. "
             "A ball partly out of the image is not measured: the last measurement is kept.")

    rep.section("6. Camera latency: without and with compensation by prediction")
    rows = []
    for ms, without, with_ in ((20, "camera_uncompensated", "camera"),
                               (60, "latency_60ms_uncompensated", "latency_60ms"),
                               (100, "latency_100ms_uncompensated", "latency_100ms")):
        for compensated, case in ((False, without), (True, with_)):
            rs = runs[case]
            if compensated:
                d_return = fmt_delta([return_distance(r) for r in runs[without]],
                                     [return_distance(r) for r in rs], " mm", scale=1000)
                d_error = fmt_delta([circle_error(r) for r in runs[without]], [circle_error(r) for r in rs],
                                    " mm", scale=1000)
            else:
                d_return = d_error = "-"
            rows.append([f"{ms} ms", "with" if compensated else "without", fallen(rs),
                         fmt([return_distance(r) for r in rs], " mm", scale=1000), d_return,
                         fmt([circle_error(r) for r in rs], " mm", scale=1000), d_error])
    rep.table(["Latency", "Compensation", "Fallen", "Max distance between 2 s and 4 s",
               "Δ (with - without)", "Circle error", "Δ circle error (with - without)"], rows)
    rows = []
    for ms, always, brain in ((20, "camera_fast_always", "camera"), (100, "latency_100ms_fast_always", "latency_100ms")):
        for label, case in (("whatever the speed of the ball", always),
                            (f"only above {BASE.fast_speed} m/s (the brain)", brain)):
            rs = runs[case]
            delta = "-" if case == always else fmt_delta([return_distance(r) for r in runs[always]],
                                                         [return_distance(r) for r in rs], " mm", scale=1000)
            rows.append([f"{ms} ms", label, fallen(rs), fmt([return_distance(r) for r in rs], " mm", scale=1000), delta,
                         fmt([circle_error(r) for r in rs], " mm", scale=1000), fmt([airborne(r) for r in rs], " %", 1, 100)])
    rep.table(["Latency", "Fast ramps of the motors", "Fallen", "Max distance between 2 s and 4 s", "Δ (brain - always)",
               "Circle error", "Airborne"], rows)
    rep.text("With compensation. The fast ramp of the motors throws a slow ball off the plate: 10 cm from the "
             "center, the plate under the ball accelerates downward faster than g beyond about 100 rad/s² of plate "
             "(r·θ̈ > g), about 200 rad/s² of motor. With a large latency, the ball then keeps bouncing; the brain "
             "keeps the fast ramps for a fast ball (thrown, pushed), which needs them.")

    rep.section("7. Integral action: error integrated everywhere or only near the setpoint")
    zone_mm = 1000 * BASE.integration_zone
    everywhere = runs["camera_integral_everywhere"]
    rows = []
    for label, case in (("everywhere", "camera_integral_everywhere"), (f"within {zone_mm:.0f} mm", "camera")):
        rs = runs[case]
        if case == "camera":
            d_return = fmt_delta([return_distance(r) for r in everywhere], [return_distance(r) for r in rs],
                                 " mm", scale=1000)
            d_error = fmt_delta([circle_error(r) for r in everywhere], [circle_error(r) for r in rs],
                                " mm", scale=1000)
        else:
            d_return = d_error = "-"
        rows.append([label, fallen(rs), fmt([arrival(r) for r in rs], " s"),
                     fmt([overshoot_after_arrival(r) for r in rs], " mm", scale=1000),
                     fmt([return_distance(r) for r in rs], " mm", scale=1000), d_return,
                     fmt([circle_error(r) for r in rs], " mm", scale=1000), d_error])
    rep.table(["Error integrated", "Fallen", "Arrival (first time < 5 mm)", "Overshoot after arrival",
               "Max distance between 2 s and 4 s", "Δ (zone - everywhere)", "Circle error",
               "Δ circle error (zone - everywhere)"], rows)
    rep.text(f"Reference scenario, latency 20 ms with compensation. With the zone, each axis integrates its "
             f"error only while it is below {zone_mm:.0f} mm: the integral no longer builds up during the "
             "approach, and only corrects the calibration error of the plate. What it still builds up over the "
             f"last {zone_mm:.0f} mm of the approach (a few tenths of a degree) is about what the nominal "
             "calibration error requires; without calibration error, it is in excess: the ball overshoots the "
             "center and comes back as the integral empties, which makes this case the worst return to the center "
             "(section 4).")

    rep.section("8. Gains of the PID: hand-tuned or by LQR")
    hand = {key: [circle_error(r) for r in runs[case]] for key, case in (("20", "camera_hand_gains"),
                                                                           ("60", "latency_60ms_hand_gains"))}
    rows = []
    for label, gains, cases, calibration, limit in (
            ("hand-tuned", REFERENCE_GAINS, ("camera_hand_gains", "latency_60ms_hand_gains"), CALIBRATION_HAND_GAINS,
             "throw_10deg_hand_gains"),
            ("LQR", (kp, ki, kd), ("camera", "latency_60ms"), CALIBRATION, "throw_10deg")):
        worst, back = worst_return(runs, calibration)
        errors = {key: [circle_error(r) for r in runs[case]] for key, case in zip(("20", "60"), cases)}
        speeds = [math.nan if v is None else v for v, _, _ in data["limits"][limit]]
        lqr = label == "LQR"
        rows.append([label, "{:.2f}, {:.2f}, {:.2f}".format(*gains), fallen(runs[cases[0]]),
                     f"{fmt(back, ' mm', scale=1000)} ({worst})", fmt(errors["20"], " mm", scale=1000),
                     fmt_delta(hand["20"], errors["20"], " mm", scale=1000) if lqr else "-",
                     fmt(errors["60"], " mm", scale=1000),
                     fmt_delta(hand["60"], errors["60"], " mm", scale=1000) if lqr else "-",
                     fmt([jitter(r) for r in runs[cases[0]]], "°", scale=180 / math.pi), fmt(speeds, " m/s")])
    rep.table(["Gains", "Kp, Ki, Kd", "Fallen", "Max distance between 2 s and 4 s (worst calibration error)",
               "Circle error", "Δ (LQR - hand-tuned)", "Circle error, 60 ms", "Δ (LQR - hand-tuned)", "θx jitter",
               "Thrown ball caught (±10°)"], rows)
    rep.text(f"Latency 20 ms with compensation, integration zone, unless stated. LQR gains (controller/lqr.py, "
             f"Bryson's rule): a position error of {100 * BASE.lqr_position:.0f} cm, an integral of the error over "
             f"{BASE.lqr_integral_time:.1f} s and a tilt of {math.degrees(BASE.lqr_tilt):.0f}° cost as much as one "
             "another. The stiffer loop corrects the calibration error of the plate faster.")

    rep.section("9. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms")
    without = [square_error(r) for r in runs["square_uncompensated"]]
    with_ = [square_error(r) for r in runs["square"]]
    rep.table(["Compensation", "Fallen", "Error (2nd lap)", "Δ (with - without)"], [
        ["without", fallen(runs["square_uncompensated"]), fmt(without, " mm", scale=1000), "-"],
        ["with", fallen(runs["square"]), fmt(with_, " mm", scale=1000),
         fmt_delta(without, with_, " mm", scale=1000)]])

    rep.section("10. Simple physics → MuJoCo: the gap, term by term (synthetic sensor, reference controller)")
    tau_ms = 1000 * PM.kv_servo / PM.kp_servo
    additions = (("+ r·θ̈ (the plate turns under the ball)", "rolling"),
                 ("+ rotating-frame terms (Euler, centrifugal, Coriolis)", "frame"),
                 ("+ exact projection of gravity", "gravity"),
                 (f"+ servo lag (τ = kv/kp = {tau_ms:.0f} ms)", "servo"),
                 ("+ all of the above", "full"))
    for level, name in (("Angle-driven plate", "plate"), ("3-arm robot", "robot")):
        basic, mujoco = f"simple_{name}", f"mujoco_{name}"
        models = (("Simple physics, basic model", basic),
                  *((label, f"simple_{name}_{suffix}") for label, suffix in additions), ("MuJoCo", mujoco))
        rows = {case: [label, fallen(runs[case])] for label, case in models}
        for indicator in (circle_error, return_distance):
            base = [indicator(r) for r in runs[basic]]
            gap = paired_delta(base, [indicator(r) for r in runs[mujoco]])[0]
            for _, case in models:
                values = [indicator(r) for r in runs[case]]
                delta = paired_delta(base, values)[0]
                share = "-" if case == basic or math.isnan(gap) or gap == 0 else f"{100 * delta / gap:.0f} %"
                rows[case] += [fmt(values, " mm", scale=1000),
                               "-" if case == basic else fmt_delta(base, values, " mm", scale=1000), share]
        rep.text(f"{level}:")
        rep.table(["Model", "Fallen", "Circle error", "Δ vs basic model", "Share of the MuJoCo gap",
                   "Max distance between 2 s and 4 s", "Δ vs basic model", "Share of the MuJoCo gap"],
                  list(rows.values()))
    rep.text("Each addition is switched on alone in the simple physics (simulation/simple_physics.py). "
             "A gap is explained when the share predicted by a model is within 80 to 120 %. The additions do not "
             "add up: on the return to the center, the rotating-frame terms and the servo lag act in opposite "
             "directions.")

    rep.section("11. Adaptive control: the model gain c·g estimated online (latency compensation, camera)")
    rows = []
    for ball, k, latency_ms, fixed, adaptive in (
            ("hollow (k = 2/3)", K, 20, "camera_fixed_gain", "camera"),
            ("hollow (k = 2/3)", K, 60, "latency_60ms_fixed_gain", "latency_60ms"),
            ("solid (k = 2/5)", 2 / 5, 20, "solid_ball_fixed_gain", "solid_ball")):
        for estimated, case in ((False, fixed), (True, adaptive)):
            rs = runs[case]
            if estimated:
                d_error = fmt_delta([circle_error(r) for r in runs[fixed]], [circle_error(r) for r in rs],
                                    " mm", scale=1000)
                d_return = fmt_delta([return_distance(r) for r in runs[fixed]], [return_distance(r) for r in rs],
                                     " mm", scale=1000)
            else:
                d_error = d_return = "-"
            rows.append([f"{ball}, {latency_ms} ms", "estimated" if estimated else "fixed",
                         fallen(rs), f"{G / (1 + k):.2f}", fmt([final_model_gain(r) for r in rs]),
                         fmt([circle_error(r) for r in rs], " mm", scale=1000), d_error,
                         fmt([return_distance(r) for r in rs], " mm", scale=1000), d_return])
    rep.table(["Ball, latency", "Model gain", "Fallen", "g/(1 + k)", f"c·g used at {BASE.duration:.0f} s",
               "Circle error", "Δ (estimated - fixed)", "Max distance between 2 s and 4 s",
               "Δ (estimated - fixed)"], rows)
    rep.text("c·g in (m/s²)/rad; fixed: the value of the hollow ball (nominal). The controller estimates c·g from "
             "the measured accelerations of the ball and the commands it sent (controller/adaptation.py): least "
             "squares, recursive, on low-pass filtered signals, from the nominal value at every attempt. It uses "
             "them only while the ball rolls (at least 3 cm/s) within 1 cm of its setpoint for at least 1 s: "
             "elsewhere the linear model does not hold (near rest; during a transient, the plate lags its "
             "command). In the reference scenario, only the circle (from 8 s) excites it: the return to the center "
             "(before 4 s) does not depend on it. The estimate replaces the nominal c·g in the feedforward and in "
             "the latency predictor; the gains of the PID stay those of the nominal model.")

    write_gaps(rep, runs, data)
    write_brains(rep, runs, data)
    rep.save(path)
    print(f"\nReport written to {path}")


def share(reproduced: float, measured: float) -> str:
    """Share of a measured gap that a model reproduces."""
    return "-" if math.isnan(reproduced) or math.isnan(measured) or measured == 0 else f"{100 * reproduced / measured:.0f} %"


def write_gaps(rep: Report, runs: dict, data: dict) -> None:
    """Section 12: every gap of more than 0.5 mm left by sections 0 to 11, explained."""
    rep.section("12. The remaining gaps of more than 0.5 mm, reproduced by our equations or traced to their cause")
    rep.text("Our equations: the simple physics of the 3-arm robot with the full equation of section 10 and sliding "
             "(Coulomb friction μ: the ball slides when rolling would need more). A gap is explained when a model "
             "reproduces it within 80 to 120 %, as in section 10.")

    def gap(base, case, indicator=return_distance):
        a, b = [indicator(r) for r in runs[base]], [indicator(r) for r in runs[case]]
        return fmt_delta(a, b, " mm", scale=1000), paired_delta(a, b)[0]

    rows = []
    for label, name in (("Solid ball (k = 2/5)", "solid_ball"), ("No calibration error (0°, 0°)", "no_calibration_error"),
                        ("Calibration error ×2 (1°, -0.6°)", "calibration_error_x2"), ("Friction μ = 0.02", "friction_0.02")):
        text_mj, mj = gap("mujoco_noise_free", f"mujoco_noise_free_{name}")
        text_eq, eq = gap("equations", f"equations_{name}")
        text_cam, cam = gap("camera", name)
        rows.append([label, text_mj, text_eq, share(eq, mj), text_cam, f"{1000 * (cam - mj):+.2f} mm"])
    text_py, py = gap("mujoco_noise_free_pyramidal", "mujoco_noise_free_pyramidal_friction_0.02")
    text_eq, eq = gap("equations", "equations_friction_0.02")
    rows.append(["Friction μ = 0.02, MuJoCo with its own friction cone (pyramidal, used until phase B)", text_py,
                 text_eq, share(eq, py), "-", "-"])
    rep.table(["Robustness variant (section 4): gap of the return to the center to the nominal case",
               "MuJoCo, sensor without noise", "Our equations, sensor without noise", "Share reproduced",
               "MuJoCo + camera (section 4)", "Effect of the camera"], rows)
    rep.text("The brain in both. The return to the center is the tail of the approach: what the integral has built up "
             "over the last 2 cm (section 7), against what the calibration error of the plate needs. A solid ball "
             "(c·g 19 % higher than the gains of the PID assume) and a plate without calibration error leave the "
             "integral in excess; a calibration error ×2 leaves it short, and the ball arrives later. With μ = 0.02, "
             "the ball slides while the plate brakes it (rolling at 10° needs μ ≥ 0.071, section 1): its path to the "
             "center changes, and the integral of one axis builds up more over the last 2 cm. The simulator uses "
             "Coulomb's round friction cone; MuJoCo's own default, a pyramid, allows up to 29 % less friction than "
             "μ·N between its axes (1 - 1/√2): with it, the ball slides more and the gap is larger (last row). "
             "Last column: the effect of the camera on each gap (MuJoCo + camera minus MuJoCo "
             "without noise).")

    rows = []
    for label, indicator in (("Max distance between 2 s and 4 s", return_distance), ("Circle error", circle_error)):
        text_mj, mj = gap("mujoco_robot", "mujoco_robot_noise_free", indicator)
        text_eq, eq = gap("simple_robot_full", "simple_robot_full_noise_free", indicator)
        rows.append([label, text_mj, text_eq, share(eq, mj)])
    rep.table(["Effect of the noise of the synthetic sensor (section 0): noise removed", "MuJoCo", "Our equations",
               "Share reproduced"], rows)
    rep.text(f"Reference controller of section 0, over {DRAWS_FACTOR['mujoco_robot']} times more draws: the "
             "noise makes the return vary by more than 1 mm from one draw to the next. Our equations reproduce the "
             "effect: it does not come from MuJoCo. The motors cannot follow the noisy command with their gentle "
             "ramp, and lag it.")

    text_robot, robot = gap("mechanism_plate", "mechanism_robot")
    text_double, double = gap("mechanism_plate", "mechanism_robot_double_ramps")
    rep.table(["Effect of the mechanism on the return (section 0), sensor without noise", "Δ vs angle-driven plate",
               "Share explained by the ramps"],
              [["3-arm robot", text_robot, "-"],
               ["3-arm robot, ramps and speed of its motors ×2", text_double,
                share(robot - double, robot)]])
    rep.text("Simple physics, reference controller, without noise to measure the effect precisely (with the noise of "
             f"section 0: {gap('simple_plate', 'simple_robot')[0]}). The motors turn about twice as much as the plate "
             "(the lever of the arms, section 0): with the same ramps, the plate of the robot tilts more slowly. Motors "
             "twice as fast remove most of the gap; the rest comes from the kinematics.")

    rows = []
    camera_return = [return_distance(r) for r in runs["latency_100ms_uncompensated"]]
    camera_circle = [circle_error(r) for r in runs["latency_100ms_uncompensated"]]
    rows.append(["MuJoCo + camera (section 6)", fmt(camera_return, " mm", scale=1000), "-",
                 fmt(camera_circle, " mm", scale=1000), "-"])
    for label, case in (("Our equations", "latency_100ms_uncompensated_equations"),
                        ("Basic model of the simple physics", "latency_100ms_uncompensated_basic")):
        ret, cir = [return_distance(r) for r in runs[case]], [circle_error(r) for r in runs[case]]
        rows.append([label, fmt(ret, " mm", scale=1000), share(mean_std(ret)[0], mean_std(camera_return)[0]),
                     fmt(cir, " mm", scale=1000), share(mean_std(cir)[0], mean_std(camera_circle)[0])])
    rep.table(["Latency 100 ms without compensation", "Max distance between 2 s and 4 s", "Share", "Circle error",
               "Share"], rows)
    rep.text(f"Synthetic sensor with the mean error of the camera (σ = {1000 * CAMERA_SIGMA:.1f} mm). Without "
             "compensation, 100 ms of latency leaves the loop lightly damped: the ball oscillates around the center "
             "for seconds. The plate turning under the ball (r·θ̈) and the lag of the servos damp it even less: the "
             "full equation reproduces MuJoCo, the basic model does not.")

    everywhere = [first_pass_minimum(r) for r in runs["camera_integral_everywhere"]]
    below = sum(v < 0.005 for v in everywhere)            # arrival: first time below 5 mm (5 % of 10 cm)
    wait, no_wait = [braking(r) for r in runs["throw_0.55_wait"]], [braking(r) for r in runs["throw_0.55_no_wait"]]
    caught = [r for v, r in sorted(data["limits"]["throw_10deg_no_wait"][0][2].items()) if not r.fallen]
    margin = 1000 * (R_PLATE - max(caught[-1].x)) if caught else math.nan
    flights = [last_flight(r) for r in runs["dropped"]]
    rep.text("Other gaps, traced to their cause:")
    rep.text(f"- Error integrated everywhere (section 7): the spread of the arrival and of the overshoot is that of a "
             f"threshold. On its first pass, the ball comes within {fmt(everywhere, ' mm', scale=1000)} of the center, "
             f"below the 5 mm of the arrival in {below} draws out of {len(everywhere)}; the arrival is then this "
             "first pass, otherwise the next one, more than a second later. The return between 2 s and 4 s does not "
             "depend on a threshold.")
    rep.text(f"- Thrown ball without waiting for its speed (section 2), at 0.55 m/s: at the first image, the position "
             f"alone tilts the plate toward the edge; it brakes the ball (tilt beyond -9.5°) at "
             f"{fmt([t for t, _ in no_wait], ' s')} instead of {fmt([t for t, _ in wait], ' s')}, when the ball is at "
             f"x = {fmt([x for _, x in no_wait], ' mm', scale=1000)} instead of {fmt([x for _, x in wait], ' mm', scale=1000)}. "
             f"Near the threshold, the ball stops at the edge: on draw 0, the fastest throw caught stops {margin:.1f} mm "
             "from it. The ball also hops a little while the plate brakes it (in the air "
             f"{fmt([airborne(r) for r in runs['throw_0.55_no_wait']], ' %', 1, 100)} of the time at 0.55 m/s): "
             "a millimetre either way decides, hence the exceptions to monotonicity.")
    rep.text(f"- Ball dropped from 30.5 cm (section 4): the max distance between 2 s and 4 s is measured while it "
             f"still bounces (last flight above {1000 * AIRBORNE:.0f} mm at {fmt(flights, ' s')}). For this variant, "
             "the criterion is that the ball does not fall.")


# --- Figures (draw 0) ----------------------------------------------------------------------------------
def write_brains(rep: Report, runs: dict, data: dict) -> None:
    """Section 13: the AI brain next to the PID brain."""
    rep.section(f"13. The AI brain (learned policy \"{AI}\") next to the PID brain (full chain with the camera)")
    rep.table(["Objective", "PID brain", "Met", "AI brain", "Met", "Target"],
              [[name, pid, pid_met, ai, ai_met, target] for (name, pid, target, pid_met), (_, ai, _, ai_met)
               in zip(objectives(data), objectives(data, "_ai"))])
    rows = []
    for label, name in (("Thrown ball (±10°), share of the physical limit", "throw_10deg"),
                        ("Push at the center, share of the physical limit", "push")):
        pid, ai = ([math.nan if v is None else v / physical_limit(name) for v, _, _ in data["limits"][name + suffix]]
                   for suffix in ("", "_ai"))
        rows.append([label, "-", fmt(pid, " %", 0, 100), "-", fmt(ai, " %", 0, 100), fmt_delta(pid, ai, " %", 1, 100)])
    cases = {"Nominal": "camera", "Latency 60 ms": "latency_60ms", "Latency 100 ms": "latency_100ms",
             **{label: case for label, case in ROBUSTNESS.items() if case != "camera"}}
    for label, case in cases.items():
        for what, indicator in (("return", return_distance), ("circle", circle_error)):
            pid, ai = ([indicator(r) for r in runs[case + suffix]] for suffix in ("", "_ai"))
            rows.append([f"{label}: {what}", fallen(runs[case]), fmt(pid, " mm", scale=1000), fallen(runs[case + "_ai"]),
                         fmt(ai, " mm", scale=1000), fmt_delta(pid, ai, " mm", scale=1000)])
    rep.table(["Case", "Fallen (PID)", "PID brain", "Fallen (AI)", "AI brain", "Δ AI - PID"], rows)
    tremor = {suffix: [jitter(r) for r in runs["latency_60ms" + suffix]] for suffix in ("", "_ai")}
    rep.text("Same cases, same draws, same rules for both brains (sensor, motors, maximum tilt). Return: max "
             "distance between 2 s and 4 s; circle: error from 12 s to 20 s. The AI brain is a neural network of 2 "
             "layers of 128 neurons that drives alone; it learned by imitating the PID brain (training/imitate.py), "
             "on the synthetic sensor, never with the camera nor with a dropped ball (it has no estimate of the "
             "flight: the PID aims its bounces). Command at rest at the center (standard deviation of θx from 2.5 s to "
             f"4 s), 60 ms of latency: {fmt(tremor[''], '°', 3, math.degrees(1))} for the PID, "
             f"{fmt(tremor['_ai'], '°', 3, math.degrees(1))} for the AI.")


def show_figures(data: dict) -> None:
    runs = data["runs"]
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 7.5))
    for v, r in sorted(data["limits"]["throw_10deg"][0][2].items()):
        a1.plot(r.t, [x * 100 for x in r.x], label=f"{v:.2f} m/s" + (" (fallen)" if r.fallen else ""))
    a1.axhline(R_PLATE * 100, color="red", linestyle="--", linewidth=0.8)
    a1.set_xlim(0, 3)
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("x (cm)")
    a1.set_title("Thrown ball, ±10°, bisection of draw 0 (red: plate edge)")
    a1.legend(fontsize=7)
    a1.grid(True)

    for case, label in (("latency_100ms_uncompensated", "without compensation"), ("latency_100ms", "with compensation")):
        r = runs[case][0]
        a2.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)], label=label)
    a2.set_ylim(0, 40)
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Error (mm)")
    a2.set_title("Latency 100 ms: ball-setpoint distance (draw 0)")
    a2.legend(fontsize=8)
    a2.grid(True)

    for case, label in (("square_uncompensated", "without compensation"), ("square", "with compensation")):
        r = runs[case][0]
        keep = [i for i, t in enumerate(r.t) if t >= 8]
        a3.plot([r.x[i] * 100 for i in keep], [r.y[i] * 100 for i in keep], label=label)
    a3.plot([5, -5, -5, 5, 5], [5, 5, -5, -5, 5], "k--", linewidth=0.8, label="setpoint")
    a3.set_aspect("equal")
    a3.set_xlabel("x (cm)")
    a3.set_ylabel("y (cm)")
    a3.set_title("Square trajectory (2nd lap, draw 0)")
    a3.legend(fontsize=8)
    a3.grid(True)

    stats = [mean_std([1000 * circle_error(r) for r in runs[case]]) for case in ROBUSTNESS.values()]
    a4.barh(range(len(ROBUSTNESS)), [m for m, _ in stats],
            xerr=[0.0 if math.isnan(s) else s for _, s in stats], color="tab:blue", capsize=3)
    a4.set_yticks(range(len(ROBUSTNESS)))
    a4.set_yticklabels(list(ROBUSTNESS), fontsize=8)
    a4.invert_yaxis()
    a4.set_xlabel("Error on the circle (mm), mean ± std over the draws")
    a4.set_title("Robustness")
    a4.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measurement campaign of the simulated balancing robot")
    parser.add_argument("--draws", type=int, default=10, help="number of draws per case (default: 10)")
    parser.add_argument("--processes", type=int, default=None, help="number of processes (default: one per core)")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    start = time.perf_counter()
    print(f"Campaign: {args.draws} draw(s) per case...")
    data = run_campaign(args.draws, args.processes)
    print(f"Simulations done in {time.perf_counter() - start:.0f} s.")
    write_report(data)
    show_figures(data)


if __name__ == "__main__":
    main()
