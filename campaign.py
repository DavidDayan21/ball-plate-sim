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

Run (from ball-plate-sim, environment activated):
    python campaign.py        (a few minutes: about thirty simulations)
"""
import math
import os
from dataclasses import replace

import matplotlib.pyplot as plt

from simulation.loop import simulate
from simulation.config import G, Config, CameraParams, square, center
from simulation.metrics import theta_std, tracking_error_rms, settling_time_5
from simulation.mujoco_physics import rolling_acceleration, measured_restitution

BASE = Config(sensor="camera", camera=CameraParams(width=160, image_height=120))
PM = BASE.mujoco_params
K = BASE.plate.k_inertia
CG = BASE.plate.coefficient * G
R_PLATE = BASE.plate.plate_radius
THROW_START = -0.10                 # the ball is thrown from x = -10 cm toward +x
md = ["# Measurement campaign report for the simulated balancing robot\n",
      "Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.\n"]


def section(title: str) -> None:
    print(f"\n{title}")
    md.append(f"\n## {title}\n")


def table(headers, rows) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    for row in [headers] + rows:
        print("  " + "  ".join(str(x).rjust(l) if i else str(x).ljust(l) for i, (x, l) in enumerate(zip(row, widths))))
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---|" * len(headers))
    md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    md.append("")


def theoretical_v_max(theta_max: float, distance: float) -> float:
    """Maximum braking a = c·g·sin θmax over the available distance: v = √(2·a·d)."""
    return math.sqrt(2 * CG * math.sin(theta_max) * distance)


def sweep(cfg_of_v, speeds):
    """Highest speed caught: increase until the first fall."""
    best, results = None, {}
    for v in speeds:
        r = simulate(cfg_of_v(v))
        results[v] = r
        if r.fallen:
            break
        best = v
    return best, results


# --- 0. From the simple model to the full chain ------------------------------------------------
section("0. From the simple model to the full chain (reference scenario, same settings)")
levels = {
    "Simple physics: angle-driven plate": Config(physics="simple", mechanism=False),
    "Simple physics: 3-arm robot": Config(physics="simple"),
    "MuJoCo: angle-driven plate": Config(mechanism=False),
    "MuJoCo: 3-arm robot": Config(),
    "MuJoCo: 3-arm robot + camera": BASE,
}
rows = []
for name, cfg in levels.items():
    r = simulate(cfg)
    contact = f"{100 * r.airborne_fraction:.1f} %" if cfg.physics == "mujoco" else "-"
    rows.append([name, "yes" if r.fallen else "no", f"{settling_time_5(r, 4):.2f} s",
                   f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm",
                   f"{math.degrees(theta_std(r, 2.5, 4)):.2f}°", contact])
table(["Level", "Fallen", "Settling time (5 %)", "Circle error", "θx jitter", "Airborne"], rows)

# --- 1. Theoretical limits ------------------------------------------------------------------
section("1. Theoretical limits and contact calibration")
mu_10 = K * math.tan(math.radians(10)) / (1 + K)
a_roll = rolling_acceleration(K, math.radians(10), friction=1.3 * mu_10)
a_slide = rolling_acceleration(K, math.radians(10), friction=0.7 * mu_10)
table(["Quantity", "Theory", "MuJoCo"], [
    ["Minimum friction to roll at 10°", f"{mu_10:.3f}", "-"],
    ["Acceleration at 10°, friction 1.3×min (rolls)", f"{G * math.sin(math.radians(10)) / (1 + K):.3f} m/s²", f"{a_roll:.3f} m/s²"],
    ["Acceleration at 10°, friction 0.7×min (slides)",
     f"{G * (math.sin(math.radians(10)) - 0.7 * mu_10 * math.cos(math.radians(10))):.3f} m/s²", f"{a_slide:.3f} m/s²"],
    ["Restitution, drop from 30.5 cm onto a fixed plane (ITTF rule)", "0.85 to 0.91", f"{measured_restitution():.3f}"],
])

# --- 2. Thrown ball -------------------------------------------------------------------------
section("2. Ball thrown from x = -10 cm toward the opposite edge (setpoint: center)")
rows, throws = [], {}
for deg in (10, 20):
    th = math.radians(deg)
    vmax, res = sweep(lambda v: replace(BASE, theta_max=th, trajectory=center, duration=5.0,
                                           start_position=(THROW_START, 0.0), start_velocity=(v, 0.0)),
                         [round(0.3 + 0.1 * i, 1) for i in range(8)])
    throws[deg] = res
    rows.append([f"±{deg}°", f"{vmax} m/s", f"{theoretical_v_max(th, R_PLATE - THROW_START):.2f} m/s"])
table(["Max tilt", "Max speed caught", "Physical limit √(2·c·g·sin θmax·d)"], rows)

# --- 3. Push --------------------------------------------------------------------------------
section("3. Push on the ball at rest at the center (t = 1 s)")
dvmax, _ = sweep(lambda dv: replace(BASE, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                                       pushes=((1.0, 0.8 * dv, 0.6 * dv),)),
                    [round(0.2 + 0.1 * i, 1) for i in range(7)])
table(["Max impulse absorbed (Δv)", "Physical limit (d = 15 cm, ±10°)"],
        [[f"{dvmax} m/s", f"{theoretical_v_max(BASE.theta_max, R_PLATE):.2f} m/s"]])

# --- 4. Physical parameters ------------------------------------------------------------------
section("4. Robustness to physical parameters (controller unchanged, tuned for the ping-pong ball)")
variants = {
    "Nominal (ping-pong, μ = 1)": BASE,
    "Solid ball (k = 2/5)": replace(BASE, mujoco_params=replace(PM, k_actual=2 / 5)),
    "Ball 10× heavier (27 g)": replace(BASE, mujoco_params=replace(PM, ball_mass=0.027)),
    "Friction μ = 0.2": replace(BASE, mujoco_params=replace(PM, friction=0.2)),
    "Friction μ = 0.02": replace(BASE, mujoco_params=replace(PM, friction=0.02)),
    "Plate 2× heavier (0.5 kg)": replace(BASE, mujoco_params=replace(PM, plate_mass=0.5)),
}
rows, nominal, param_errors = [], None, {}
for name, cfg in variants.items():
    r = simulate(cfg)
    nominal = nominal or r
    param_errors[name] = tracking_error_rms(r, 12, 20) * 1000
    rows.append([name, "yes" if r.fallen else "no", f"{settling_time_5(r, 4):.2f} s",
                   f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm", f"{100 * r.airborne_fraction:.1f} %"])
table(["Variant", "Fallen", "Settling time (5 %)", "Circle error", "Airborne"], rows)

# --- 5. Latency and compensation -------------------------------------------------------------
section("5. Camera latency: without and with compensation by prediction")
rows, latency = [], {}
for n in (1, 3, 5):
    for comp in (False, True):
        r = nominal if (n == 1 and not comp) else simulate(replace(BASE, latency=n, latency_compensation=comp))
        latency[(n, comp)] = r
        rows.append([f"{20 * n} ms", "with" if comp else "without", "yes" if r.fallen else "no",
                       f"{settling_time_5(r, 4):.2f} s", f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm"])
table(["Latency", "Compensation", "Fallen", "Settling time (5 %)", "Circle error"], rows)

# --- 6. Square ----------------------------------------------------------------------------------
section("6. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms")
rows, squares = [], {}
for comp in (False, True):
    r = simulate(replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05), latency_compensation=comp))
    squares[comp] = r
    rows.append(["with" if comp else "without", "yes" if r.fallen else "no", f"{tracking_error_rms(r, 8, 16) * 1000:.2f} mm"])
table(["Compensation", "Fallen", "Error (2nd lap)"], rows)

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\nReport written to results/report.md")

# --- Figures -------------------------------------------------------------------------------------
fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 7.5))
for v, r in throws[10].items():
    a1.plot(r.t, [x * 100 for x in r.x], label=f"{v} m/s" + (" (fallen)" if r.fallen else ""))
a1.axhline(R_PLATE * 100, color="red", linestyle="--", linewidth=0.8)
a1.set_xlim(0, 3)
a1.set_xlabel("Time (s)")
a1.set_ylabel("x (cm)")
a1.set_title("Thrown ball, ±10° (red: plate edge)")
a1.legend(fontsize=7)
a1.grid(True)

for comp in (False, True):
    r = latency[(5, comp)]
    a2.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)],
            label="with compensation" if comp else "without compensation")
a2.set_ylim(0, 40)
a2.set_xlabel("Time (s)")
a2.set_ylabel("Error (mm)")
a2.set_title("Latency 100 ms: ball-setpoint distance")
a2.legend(fontsize=8)
a2.grid(True)

for comp, r in squares.items():
    keep = [i for i, t in enumerate(r.t) if t >= 8]
    a3.plot([r.x[i] * 100 for i in keep], [r.y[i] * 100 for i in keep],
            label="with compensation" if comp else "without compensation")
a3.plot([5, -5, -5, 5, 5], [5, 5, -5, -5, 5], "k--", linewidth=0.8, label="setpoint")
a3.set_aspect("equal")
a3.set_xlabel("x (cm)")
a3.set_ylabel("y (cm)")
a3.set_title("Square trajectory (2nd lap)")
a3.legend(fontsize=8)
a3.grid(True)

names = list(param_errors)
a4.barh(range(len(names)), [param_errors[n] for n in names], color="tab:blue")
a4.set_yticks(range(len(names)))
a4.set_yticklabels(names, fontsize=8)
a4.invert_yaxis()
a4.set_xlabel("Error on the circle (mm)")
a4.set_title("Robustness to physical parameters")
a4.grid(True, axis="x")
plt.tight_layout()
plt.show()
