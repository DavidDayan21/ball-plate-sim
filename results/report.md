# Measurement campaign report for the simulated balancing robot

Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.

Each case is simulated over 10 draw(s). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise seed k and a start position shifted by up to ±0.5 mm on each axis (the only source of variation for the camera cases). Values: mean ± standard deviation over the draws; draws where the ball fell are counted in 'Fallen' and left out of the other columns. Δ: mean of the paired differences over the same draws ± standard error, in **bold** when significant (paired t-test at the 5 % level: |Δ| > t × standard error, t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws).

The brain is the PID with its latency compensation and its integration zone (default settings): it is judged by the objectives and sections 2 to 7. Sections 0 and 8 validate the simulator: they use a frozen reference controller (the PID without latency compensation, error integrated everywhere).


## Objectives of the brain (full chain with the camera)

| Objective | Measured | Target | Met |
|---|---|---|---|
| Thrown ball caught (±10°) | 71 ± 1 % | ≥ 90 % of the physical limit | **no** |
| Circle with 60 ms of latency | 1.66 ± 0.02 mm | < 2 mm | yes |
| Back to the center in less than 2 s | 12.93 ± 0.07 mm (calibration error: ×2) | ≤ 5 mm between 2 s and 4 s | **no** |
| No fall in the robustness variants | 0/80 falls | 0 | yes |

Judged on the mean over the draws. Thrown ball: maximum speed caught at ±10° divided by the physical limit (section 2). Return: start 10 cm away, setpoint at the center, worst of the three calibration errors of the plate of section 4 (none, nominal 0.5°/-0.3°, ×2), since the actual one is unknown. Robustness: every variant and every draw of section 4.


## 0. From the simple model to the full chain (reference scenario, reference controller)

| Level | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s |
|---|---|---|---|---|
| Simple physics: angle-driven plate | 0/10 | 0.81 ± 0.01 s | 36.98 ± 0.59 mm | 28.26 ± 0.48 mm |
| Simple physics: 3-arm robot | 0/10 | 0.82 ± 0.01 s | 40.27 ± 0.88 mm | 29.75 ± 1.13 mm |
| MuJoCo: angle-driven plate | 0/10 | 0.86 ± 0.02 s | 37.60 ± 1.02 mm | 32.90 ± 1.04 mm |
| MuJoCo: 3-arm robot | 0/10 | 0.81 ± 0.01 s | 40.08 ± 1.02 mm | 29.98 ± 1.09 mm |
| MuJoCo: 3-arm robot + camera | 0/10 | 0.82 ± 0.00 s | 40.70 ± 0.13 mm | 30.80 ± 0.12 mm |

Return to the center (start 10 cm away, setpoint at the center until 4 s). Objective "back to the center in less than 2 s": max distance between 2 s and 4 s ≤ 5 mm.

| Level | Circle error | Factor changed | Δ circle error (effect of the factor) | θx jitter | Airborne |
|---|---|---|---|---|---|
| Simple physics: angle-driven plate | 3.43 ± 0.08 mm | - | - | 0.58 ± 0.05° | - |
| Simple physics: 3-arm robot | 4.75 ± 0.45 mm | mechanism | **+1.32 ± 0.15 mm** | 0.59 ± 0.06° | - |
| MuJoCo: angle-driven plate | 2.56 ± 0.71 mm | physics | **-0.87 ± 0.23 mm** | 0.58 ± 0.05° | 72.3 ± 2.9 % |
| MuJoCo: 3-arm robot | 5.39 ± 0.45 mm | physics | **+0.64 ± 0.09 mm** | 0.59 ± 0.06° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot + camera | 4.61 ± 0.03 mm | sensor | **-0.79 ± 0.14 mm** | 0.12 ± 0.00° | 0.0 ± 0.0 % |

Circle: 6 cm radius in 4 s, error = root-mean-square distance from 12 s to 20 s. Factor changed: compared with the level above that differs only by this factor (the mechanism: the simple plate; the physics: the same level in simple physics; the sensor: the MuJoCo robot).


## 1. Theoretical limits and contact calibration

| Quantity | Theory | MuJoCo |
|---|---|---|
| Minimum friction to roll at 10° | 0.071 | - |
| Acceleration at 10°, friction 1.3×min (rolls) | 1.022 m/s² | 1.022 m/s² |
| Acceleration at 10°, friction 0.7×min (slides) | 1.227 m/s² | 1.234 m/s² |
| Restitution, drop from 30.5 cm onto a fixed plane (ITTF rule) | 0.85 to 0.91 | 0.874 |


## 2. Ball thrown from x = -10 cm toward the opposite edge (setpoint: center)

| Max tilt | Max speed caught | Physical limit √(2·c·g·sin θmax·d) | Share of the limit | Monotonic around the threshold (draw 0) |
|---|---|---|---|---|
| ±10° | 0.51 ± 0.01 m/s | 0.71 m/s | 71 ± 1 % | yes |
| ±20° | 0.62 ± 0.00 m/s | 1.00 m/s | 62 ± 0 % | yes |


## 3. Push on the ball at rest at the center (t = 1 s)

| Max tilt | Max impulse absorbed (Δv) | Physical limit √(2·c·g·sin θmax·d) | Share of the limit | Monotonic around the threshold (draw 0) |
|---|---|---|---|---|
| ±10° | 0.45 ± 0.00 m/s | 0.55 m/s | 81 ± 0 % | yes |

Maximum speeds found by bisection, to 0.01 m/s, between 0.1 and 1.5 m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).


## 4. Robustness to physical parameters (controller unchanged, tuned for the ping-pong ball)

| Variant | Fallen | Max distance between 2 s and 4 s | Circle error | Δ circle error vs nominal | Airborne |
|---|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | 0/10 | 2.68 ± 0.02 mm | 1.49 ± 0.01 mm | - | 0.0 ± 0.0 % |
| Solid ball (k = 2/5) | 0/10 | 2.80 ± 0.11 mm | 5.56 ± 0.01 mm | **+4.07 ± 0.00 mm** | 0.0 ± 0.0 % |
| Ball 10× heavier (27 g) | 0/10 | 2.61 ± 0.04 mm | 1.47 ± 0.02 mm | **-0.02 ± 0.01 mm** | 0.0 ± 0.0 % |
| Friction μ = 0.2 | 0/10 | 2.61 ± 0.05 mm | 1.42 ± 0.02 mm | **-0.07 ± 0.01 mm** | 0.0 ± 0.0 % |
| Friction μ = 0.02 | 0/10 | 2.97 ± 0.03 mm | 1.40 ± 0.01 mm | **-0.10 ± 0.01 mm** | 0.0 ± 0.0 % |
| Plate 2× heavier (0.5 kg) | 0/10 | 2.65 ± 0.03 mm | 1.50 ± 0.02 mm | +0.01 ± 0.01 mm | 0.0 ± 0.0 % |
| No calibration error (0°, 0°) | 0/10 | 6.23 ± 0.20 mm | 1.51 ± 0.02 mm | +0.01 ± 0.01 mm | 0.0 ± 0.0 % |
| Calibration error ×2 (1°, -0.6°) | 0/10 | 12.93 ± 0.07 mm | 1.48 ± 0.01 mm | **-0.02 ± 0.00 mm** | 0.0 ± 0.0 % |


## 5. Camera latency: without and with compensation by prediction

| Latency | Compensation | Fallen | Max distance between 2 s and 4 s | Δ (with - without) | Circle error | Δ circle error (with - without) |
|---|---|---|---|---|---|---|
| 20 ms | without | 0/10 | 2.48 ± 0.06 mm | - | 4.59 ± 0.04 mm | - |
| 20 ms | with | 0/10 | 2.68 ± 0.02 mm | **+0.20 ± 0.02 mm** | 1.49 ± 0.01 mm | **-3.10 ± 0.01 mm** |
| 60 ms | without | 0/10 | 2.04 ± 0.07 mm | - | 10.59 ± 0.03 mm | - |
| 60 ms | with | 0/10 | 2.81 ± 0.18 mm | **+0.77 ± 0.07 mm** | 1.66 ± 0.02 mm | **-8.93 ± 0.01 mm** |
| 100 ms | without | 0/10 | 2.47 ± 0.06 mm | - | 16.68 ± 0.04 mm | - |
| 100 ms | with | 0/10 | 3.01 ± 0.05 mm | **+0.53 ± 0.03 mm** | 1.94 ± 0.03 mm | **-14.74 ± 0.02 mm** |


## 6. Integral action: error integrated everywhere or only near the setpoint

| Error integrated | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Δ (zone - everywhere) | Circle error | Δ circle error (zone - everywhere) |
|---|---|---|---|---|---|---|---|
| everywhere | 0/10 | 0.86 ± 0.00 s | 39.78 ± 0.09 mm | 32.13 ± 0.11 mm | - | 1.46 ± 0.02 mm | - |
| within 20 mm | 0/10 | 1.70 ± 0.00 s | 4.84 ± 0.03 mm | 2.68 ± 0.02 mm | **-29.45 ± 0.04 mm** | 1.49 ± 0.01 mm | **+0.03 ± 0.01 mm** |

Reference scenario, latency 20 ms with compensation. With the zone, each axis integrates its error only while it is below 20 mm: the integral no longer builds up during the approach, and only corrects the calibration error of the plate.


## 7. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms

| Compensation | Fallen | Error (2nd lap) | Δ (with - without) |
|---|---|---|---|
| without | 0/10 | 9.83 ± 0.01 mm | - |
| with | 0/10 | 9.34 ± 0.01 mm | **-0.49 ± 0.01 mm** |


## 8. Simple physics → MuJoCo on the 3-arm robot: the gap, term by term (synthetic sensor, reference controller)

| Model | Fallen | Circle error | Δ vs basic model | Share of the MuJoCo gap |
|---|---|---|---|---|
| Simple physics, basic model | 0/10 | 4.75 ± 0.45 mm | - | - |
| + r·θ̈ (the plate turns under the ball) | 0/10 | 4.90 ± 0.42 mm | +0.15 ± 0.12 mm | 23 % |
| + rotating-frame terms (Euler, centrifugal, Coriolis) | 0/10 | 4.79 ± 0.39 mm | +0.04 ± 0.05 mm | 6 % |
| + exact projection of gravity | 0/10 | 4.73 ± 0.32 mm | -0.02 ± 0.08 mm | -3 % |
| + servo lag (τ = kv/kp = 10 ms) | 0/10 | 5.33 ± 0.29 mm | **+0.58 ± 0.08 mm** | 90 % |
| + all of the above | 0/10 | 5.34 ± 0.37 mm | **+0.59 ± 0.13 mm** | 92 % |
| MuJoCo | 0/10 | 5.39 ± 0.45 mm | **+0.64 ± 0.09 mm** | 100 % |

Each addition is switched on alone in the simple physics (simulation/simple_physics.py). A gap is explained when the share predicted by a model is within 80 to 120 %.

