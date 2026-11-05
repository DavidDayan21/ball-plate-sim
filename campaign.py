"""Measurement campaign: consistency of the models, robustness, latency. Writes results/report.md.

Reference chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.
The 160×120 camera is as accurate as 320×240, and faster to simulate.

    0. From the simple model to the full chain: what each level of realism adds
    1. Theoretical limits: no-slip condition; bounce compliant with the ITTF rule
    2. Thrown ball: maximum speed caught (max tilt 10°, then 20°)
    3. Push on a ball at rest at the center: maximum impulse absorbed
    4. Robustness to physical parameters
    5. Camera latency, with and without compensation by prediction
    6. Square trajectory
    7. Gap between the simple physics and MuJoCo on the robot, explained term by term

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

from simulation.config import G, Config, CameraParams, square, center
from simulation.loop import simulate
from simulation.metrics import arrival_time, max_distance, overshoot, theta_std, tracking_error_rms
from simulation.mujoco_physics import rolling_acceleration, measured_restitution
from simulation.parallel import execute

BASE = Config(sensor="camera", camera=CameraParams(width=160, image_height=120))
PM = BASE.mujoco_params
K = BASE.plate.k_inertia
CG = BASE.plate.coefficient * G
R_PLATE = BASE.plate.plate_radius
THROW_START = -0.10                 # the ball is thrown from x = -10 cm toward +x
START_OFFSET = 0.0005               # (m) draws k ≥ 1: start position shifted by up to ±0.5 mm per axis
SPEED_STEP = 0.01                   # (m/s) resolution of the maximum speeds
SPEED_RANGE = (0.1, 1.5)            # (m/s) bisection bracket: caught at the low end, fallen at the high end
CHECK_SPAN = 5                      # draw 0: every speed within ±5 steps of the threshold is simulated

# Closed-loop cases (nominal configurations), each simulated over every draw
CASES = {
    "simple_plate": Config(physics="simple", mechanism=False),
    "simple_robot": Config(physics="simple"),
    "simple_robot_rolling": Config(physics="simple", rolling_term=True),
    "simple_robot_frame": Config(physics="simple", frame_terms=True),
    "simple_robot_gravity": Config(physics="simple", exact_gravity=True),
    "simple_robot_servo": Config(physics="simple", servo_lag=True),
    "simple_robot_full": Config(physics="simple", rolling_term=True, frame_terms=True, exact_gravity=True,
                                servo_lag=True),
    "mujoco_plate": Config(mechanism=False),
    "mujoco_robot": Config(),
    "camera": BASE,
    "solid_ball": replace(BASE, mujoco_params=replace(PM, k_actual=2 / 5)),
    "heavy_ball": replace(BASE, mujoco_params=replace(PM, ball_mass=0.027)),
    "friction_0.2": replace(BASE, mujoco_params=replace(PM, friction=0.2)),
    "friction_0.02": replace(BASE, mujoco_params=replace(PM, friction=0.02)),
    "heavy_plate": replace(BASE, mujoco_params=replace(PM, plate_mass=0.5)),
    "latency_20ms_compensated": replace(BASE, latency_compensation=True),
    "latency_60ms": replace(BASE, latency=3),
    "latency_60ms_compensated": replace(BASE, latency=3, latency_compensation=True),
    "latency_100ms": replace(BASE, latency=5),
    "latency_100ms_compensated": replace(BASE, latency=5, latency_compensation=True),
    "square": replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05)),
    "square_compensated": replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05),
                                  latency_compensation=True),
}
# Maximum speeds searched by bisection: name -> (kind, max tilt in degrees)
LIMITS = {"throw_10deg": ("throw", 10), "throw_20deg": ("throw", 20), "push": ("push", 10)}


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


# 97.5 % quantiles of Student's t distribution, for 1 to 30 degrees of freedom (1.96 beyond)
T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


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
    kind, tilt_deg = LIMITS[name]
    if kind == "throw":
        return replace(BASE, theta_max=math.radians(tilt_deg), trajectory=center, duration=5.0,
                       start_position=(THROW_START, 0.0), start_velocity=(v, 0.0))
    return replace(BASE, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                   pushes=((1.0, 0.8 * v, 0.6 * v),))


def physical_limit(name: str) -> float:
    """Maximum braking a = c·g·sin θmax over the available distance d: v = √(2·a·d)."""
    kind, tilt_deg = LIMITS[name]
    distance = R_PLATE - THROW_START if kind == "throw" else R_PLATE
    return math.sqrt(2 * CG * math.sin(math.radians(tilt_deg)) * distance)


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


def jitter(r) -> float:
    return math.nan if r.fallen else theta_std(r, 2.5, 4)


def airborne(r) -> float:
    return math.nan if r.fallen else r.airborne_fraction


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
    jobs = [(name, k) for name in CASES for k in range(draws)]
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
             "t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws).")

    rep.section("0. From the simple model to the full chain (reference scenario, same settings)")
    # Each level is compared with the level that differs from it by a single factor
    levels = (("Simple physics: angle-driven plate", "simple_plate", None, "-"),
              ("Simple physics: 3-arm robot", "simple_robot", "simple_plate", "mechanism"),
              ("MuJoCo: angle-driven plate", "mujoco_plate", "simple_plate", "physics"),
              ("MuJoCo: 3-arm robot", "mujoco_robot", "simple_robot", "physics"),
              ("MuJoCo: 3-arm robot + camera", "camera", "mujoco_robot", "sensor"))
    rows = []
    for label, case, _, _ in levels:
        rs = runs[case]
        rows.append([label, fallen(rs), fmt([arrival(r) for r in rs], " s"),
                     fmt([overshoot_after_arrival(r) for r in rs], " mm", scale=1000),
                     fmt([return_distance(r) for r in rs], " mm", scale=1000)])
    rep.table(["Level", "Fallen", "Arrival (first time < 5 mm)", "Overshoot after arrival",
               "Max distance between 2 s and 4 s"], rows)
    rep.text("Return to the center (start 10 cm away, setpoint at the center until 4 s). Objective "
             "\"back to the center in less than 2 s\": max distance between 2 s and 4 s ≤ 5 mm.")
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
             "(the mechanism: the simple plate; the physics: the same level in simple physics; "
             "the sensor: the MuJoCo robot).")

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
              [limit_row("throw_10deg", "±10°"), limit_row("throw_20deg", "±20°")])
    rep.section("3. Push on the ball at rest at the center (t = 1 s)")
    rep.table(["Max tilt", "Max impulse absorbed (Δv)"] + headers, [limit_row("push", "±10°")])
    rep.text(f"Maximum speeds found by bisection, to {SPEED_STEP:.2f} m/s, between {SPEED_RANGE[0]} and "
             f"{SPEED_RANGE[1]} m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).")

    rep.section("4. Robustness to physical parameters (controller unchanged, tuned for the ping-pong ball)")
    variants = {"Nominal (ping-pong, μ = 1)": "camera", "Solid ball (k = 2/5)": "solid_ball",
                "Ball 10× heavier (27 g)": "heavy_ball", "Friction μ = 0.2": "friction_0.2",
                "Friction μ = 0.02": "friction_0.02", "Plate 2× heavier (0.5 kg)": "heavy_plate"}
    nominal = [circle_error(r) for r in runs["camera"]]
    rows = []
    for label, case in variants.items():
        rs = runs[case]
        errors = [circle_error(r) for r in rs]
        rows.append([label, fallen(rs), fmt([return_distance(r) for r in rs], " mm", scale=1000),
                     fmt(errors, " mm", scale=1000),
                     "-" if case == "camera" else fmt_delta(nominal, errors, " mm", scale=1000),
                     fmt([airborne(r) for r in rs], " %", 1, 100)])
    rep.table(["Variant", "Fallen", "Max distance between 2 s and 4 s", "Circle error",
               "Δ circle error vs nominal", "Airborne"], rows)

    rep.section("5. Camera latency: without and with compensation by prediction")
    rows = []
    for ms, without, with_ in ((20, "camera", "latency_20ms_compensated"),
                               (60, "latency_60ms", "latency_60ms_compensated"),
                               (100, "latency_100ms", "latency_100ms_compensated")):
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

    rep.section("6. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms")
    without = [square_error(r) for r in runs["square"]]
    with_ = [square_error(r) for r in runs["square_compensated"]]
    rep.table(["Compensation", "Fallen", "Error (2nd lap)", "Δ (with - without)"], [
        ["without", fallen(runs["square"]), fmt(without, " mm", scale=1000), "-"],
        ["with", fallen(runs["square_compensated"]), fmt(with_, " mm", scale=1000),
         fmt_delta(without, with_, " mm", scale=1000)]])

    rep.section("7. Simple physics → MuJoCo on the 3-arm robot: the gap, term by term (synthetic sensor)")
    base = [circle_error(r) for r in runs["simple_robot"]]
    gap = paired_delta(base, [circle_error(r) for r in runs["mujoco_robot"]])[0]
    tau_ms = 1000 * PM.kv_servo / PM.kp_servo
    models = (("Simple physics, basic model", "simple_robot"),
              ("+ r·θ̈ (the plate turns under the ball)", "simple_robot_rolling"),
              ("+ rotating-frame terms (Euler, centrifugal, Coriolis)", "simple_robot_frame"),
              ("+ exact projection of gravity", "simple_robot_gravity"),
              (f"+ servo lag (τ = kv/kp = {tau_ms:.0f} ms)", "simple_robot_servo"),
              ("+ all of the above", "simple_robot_full"),
              ("MuJoCo", "mujoco_robot"))
    rows = []
    for label, case in models:
        errors = [circle_error(r) for r in runs[case]]
        delta = paired_delta(base, errors)[0]
        share = "-" if case == "simple_robot" or math.isnan(gap) or gap == 0 else f"{100 * delta / gap:.0f} %"
        rows.append([label, fallen(runs[case]), fmt(errors, " mm", scale=1000),
                     "-" if case == "simple_robot" else fmt_delta(base, errors, " mm", scale=1000), share])
    rep.table(["Model", "Fallen", "Circle error", "Δ vs basic model", "Share of the MuJoCo gap"], rows)
    rep.text("Each addition is switched on alone in the simple physics (simulation/simple_physics.py). "
             "A gap is explained when the share predicted by a model is within 80 to 120 %.")

    rep.save(path)
    print(f"\nReport written to {path}")


# --- Figures (draw 0) ----------------------------------------------------------------------------------
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

    for case, label in (("latency_100ms", "without compensation"), ("latency_100ms_compensated", "with compensation")):
        r = runs[case][0]
        a2.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)], label=label)
    a2.set_ylim(0, 40)
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Error (mm)")
    a2.set_title("Latency 100 ms: ball-setpoint distance (draw 0)")
    a2.legend(fontsize=8)
    a2.grid(True)

    for case, label in (("square", "without compensation"), ("square_compensated", "with compensation")):
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

    variants = {"Nominal (ping-pong, μ = 1)": "camera", "Solid ball (k = 2/5)": "solid_ball",
                "Ball 10× heavier (27 g)": "heavy_ball", "Friction μ = 0.2": "friction_0.2",
                "Friction μ = 0.02": "friction_0.02", "Plate 2× heavier (0.5 kg)": "heavy_plate"}
    stats = [mean_std([1000 * circle_error(r) for r in runs[case]]) for case in variants.values()]
    a4.barh(range(len(variants)), [m for m, _ in stats],
            xerr=[0.0 if math.isnan(s) else s for _, s in stats], color="tab:blue", capsize=3)
    a4.set_yticks(range(len(variants)))
    a4.set_yticklabels(list(variants), fontsize=8)
    a4.invert_yaxis()
    a4.set_xlabel("Error on the circle (mm), mean ± std over the draws")
    a4.set_title("Robustness to physical parameters")
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
