# Measurement campaign report for the simulated balancing robot

Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.

Each case is simulated over 10 draw(s). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise seed k and a start position shifted by up to ±0.5 mm on each axis (the only source of variation for the camera cases). Values: mean ± standard deviation over the draws; draws where the ball fell are counted in 'Fallen' and left out of the other columns. Δ: mean of the paired differences over the same draws ± standard error, in **bold** when significant (paired t-test at the 5 % level: |Δ| > t × standard error, t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws).

The brain is the PID with its latency compensation, its integration zone and its aimed bounce (default settings): it is judged by the objectives and sections 2 to 8. Sections 0 and 9 validate the simulator: they use a frozen reference controller (the PID without latency compensation, error integrated everywhere, ball never considered in the air).


## Objectives of the brain (full chain with the camera)

| Objective | Measured | Target | Met |
|---|---|---|---|
| Thrown ball caught (±10°) | 89 ± 3 % | ≥ 90 % of the physical limit | **no** |
| Circle with 60 ms of latency | 1.65 ± 0.02 mm | < 2 mm | yes |
| Back to the center in less than 2 s | 13.61 ± 0.14 mm (calibration error: ×2) | ≤ 5 mm between 2 s and 4 s | **no** |
| No fall in the robustness variants | 0/90 falls | 0 | yes |

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
| ±10° | 0.64 ± 0.02 m/s | 0.71 m/s | 89 ± 3 % | yes |
| ±20° | 0.84 ± 0.01 m/s | 1.00 m/s | 84 ± 1 % | yes |
| ±10°, without waiting for the speed | 0.62 ± 0.01 m/s | 0.71 m/s | 87 ± 2 % | yes |
| ±10°, single motor ramp of 50 rad/s² (before B4) | 0.51 ± 0.00 m/s | 0.71 m/s | 72 ± 1 % | yes |

The brain waits for the second measurement, which gives the speed of the ball, before tilting the plate: at the first one, the position alone would tilt it the wrong way. Motor ramps: 300 rad/s² for the large moves (gap above 0.1 rad: the plate flips in about 0.1 s), 50 rad/s² for the small corrections.


## 3. Push on the ball at rest at the center (t = 1 s)

| Max tilt | Max impulse absorbed (Δv) | Physical limit √(2·c·g·sin θmax·d) | Share of the limit | Monotonic around the threshold (draw 0) |
|---|---|---|---|---|
| ±10° | 0.49 ± 0.00 m/s | 0.55 m/s | 88 ± 0 % | yes |

Maximum speeds found by bisection, to 0.01 m/s, between 0.1 and 1.5 m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).


## 4. Robustness to physical parameters and to a dropped ball (controller unchanged, tuned for the ping-pong ball)

| Variant | Fallen | Max distance between 2 s and 4 s | Circle error | Δ circle error vs nominal | Airborne |
|---|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | 0/10 | 2.84 ± 0.09 mm | 1.49 ± 0.02 mm | - | 2.5 ± 0.5 % |
| Solid ball (k = 2/5) | 0/10 | 3.11 ± 0.08 mm | 5.54 ± 0.02 mm | **+4.05 ± 0.01 mm** | 2.5 ± 0.5 % |
| Ball 10× heavier (27 g) | 0/10 | 2.81 ± 0.02 mm | 1.45 ± 0.02 mm | **-0.03 ± 0.01 mm** | 0.6 ± 0.1 % |
| Friction μ = 0.2 | 0/10 | 2.85 ± 0.07 mm | 1.41 ± 0.01 mm | **-0.08 ± 0.01 mm** | 2.2 ± 0.5 % |
| Friction μ = 0.02 | 0/10 | 3.12 ± 0.05 mm | 1.41 ± 0.02 mm | **-0.07 ± 0.01 mm** | 3.6 ± 1.2 % |
| Plate 2× heavier (0.5 kg) | 0/10 | 2.85 ± 0.05 mm | 1.45 ± 0.03 mm | **-0.04 ± 0.01 mm** | 2.2 ± 0.5 % |
| No calibration error (0°, 0°) | 0/10 | 6.52 ± 0.14 mm | 1.49 ± 0.01 mm | -0.00 ± 0.01 mm | 2.6 ± 0.3 % |
| Calibration error ×2 (1°, -0.6°) | 0/10 | 13.61 ± 0.14 mm | 1.47 ± 0.02 mm | **-0.02 ± 0.01 mm** | 2.1 ± 0.4 % |
| Ball dropped from 30.5 cm (ITTF height) | 0/10 | 15.99 ± 1.44 mm | 1.49 ± 0.01 mm | -0.00 ± 0.01 mm | 17.1 ± 0.7 % |

Reference scenario. Dropped ball: released without speed above the start point, 30.5 cm above the plate. A ball in the air beyond the edge of the tilted plate counts as fallen only once it is level with the plate.


## 5. Dropped ball: where and from how high (setpoint: center)

| Distance from the center | Height | Fallen, reacting in the air | Fallen, aimed bounce | Airborne | Max distance from the center between 4 s and 6 s | Height measured in the air: error (RMS) | On the plate but measured in the air |
|---|---|---|---|---|---|---|---|
| 0 cm | 10.0 cm | 0/10 | 0/10 | 32.4 ± 0.5 % | 0.55 ± 0.06 mm | 2.77 ± 0.47 mm | 0.0 ± 0.0 % |
| 0 cm | 20.0 cm | 0/10 | 0/10 | 46.7 ± 1.9 % | 2.32 ± 0.36 mm | 2.17 ± 0.12 mm | 0.0 ± 0.0 % |
| 0 cm | 30.5 cm | 0/10 | 0/10 | 56.7 ± 2.1 % | 3.49 ± 0.40 mm | 2.19 ± 0.13 mm | 0.0 ± 0.0 % |
| 5 cm | 10.0 cm | 0/10 | 0/10 | 32.3 ± 1.0 % | 0.59 ± 0.17 mm | 2.30 ± 0.19 mm | 0.0 ± 0.0 % |
| 5 cm | 20.0 cm | 0/10 | 0/10 | 49.5 ± 2.2 % | 3.08 ± 0.53 mm | 2.09 ± 0.12 mm | 0.0 ± 0.0 % |
| 5 cm | 30.5 cm | 10/10 | 0/10 | 55.4 ± 1.6 % | 3.61 ± 0.79 mm | 2.06 ± 0.18 mm | 0.0 ± 0.0 % |
| 10 cm | 10.0 cm | 0/10 | 0/10 | 32.6 ± 0.9 % | 1.06 ± 0.29 mm | 3.12 ± 0.23 mm | 0.0 ± 0.0 % |
| 10 cm | 20.0 cm | 10/10 | 0/10 | 45.9 ± 2.3 % | 3.03 ± 0.45 mm | 2.80 ± 0.18 mm | 0.0 ± 0.0 % |
| 10 cm | 30.5 cm | 0/10 | 0/10 | 54.9 ± 2.3 % | 4.09 ± 0.36 mm | 6.09 ± 0.15 mm | 0.0 ± 0.0 % |

Ball released without speed above a point of the line from the center to the start point of the reference scenario, 6 s per drop. Reacting in the air: the brain before B5, which drives the plate from the measured position of the ball also while it is in the air. Aimed bounce (the brain): in the air, the plate is tilted so that the next bounce sends the ball toward the setpoint (controller/flight.py, controller/plate.py); the other columns are those of the brain. Height: measured by the camera from the apparent size of the ball; in the air above 15 mm (true height), on the plate below 5 mm. A ball partly out of the image is not measured: the last measurement is kept.


## 6. Camera latency: without and with compensation by prediction

| Latency | Compensation | Fallen | Max distance between 2 s and 4 s | Δ (with - without) | Circle error | Δ circle error (with - without) |
|---|---|---|---|---|---|---|
| 20 ms | without | 0/10 | 2.66 ± 0.08 mm | - | 4.59 ± 0.02 mm | - |
| 20 ms | with | 0/10 | 2.84 ± 0.09 mm | **+0.18 ± 0.03 mm** | 1.49 ± 0.02 mm | **-3.10 ± 0.01 mm** |
| 60 ms | without | 0/10 | 2.34 ± 0.15 mm | - | 10.59 ± 0.02 mm | - |
| 60 ms | with | 0/10 | 2.90 ± 0.08 mm | **+0.57 ± 0.05 mm** | 1.65 ± 0.02 mm | **-8.94 ± 0.01 mm** |
| 100 ms | without | 0/10 | 2.79 ± 0.49 mm | - | 16.67 ± 0.04 mm | - |
| 100 ms | with | 0/10 | 3.02 ± 0.12 mm | +0.24 ± 0.16 mm | 1.92 ± 0.03 mm | **-14.76 ± 0.02 mm** |


## 7. Integral action: error integrated everywhere or only near the setpoint

| Error integrated | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Δ (zone - everywhere) | Circle error | Δ circle error (zone - everywhere) |
|---|---|---|---|---|---|---|---|
| everywhere | 0/10 | 0.88 ± 0.00 s | 37.55 ± 0.38 mm | 31.65 ± 0.10 mm | - | 1.49 ± 0.01 mm | - |
| within 20 mm | 0/10 | 1.74 ± 0.02 s | 4.89 ± 0.08 mm | 2.84 ± 0.09 mm | **-28.80 ± 0.05 mm** | 1.49 ± 0.02 mm | +0.00 ± 0.01 mm |

Reference scenario, latency 20 ms with compensation. With the zone, each axis integrates its error only while it is below 20 mm: the integral no longer builds up during the approach, and only corrects the calibration error of the plate.


## 8. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms

| Compensation | Fallen | Error (2nd lap) | Δ (with - without) |
|---|---|---|---|
| without | 0/10 | 9.84 ± 0.02 mm | - |
| with | 0/10 | 9.35 ± 0.01 mm | **-0.49 ± 0.01 mm** |


## 9. Simple physics → MuJoCo on the 3-arm robot: the gap, term by term (synthetic sensor, reference controller)

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

