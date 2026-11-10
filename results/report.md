# Measurement campaign report for the simulated balancing robot

Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.

Each case is simulated over 10 draw(s). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise seed k and a start position shifted by up to ±0.5 mm on each axis (the only source of variation for the camera cases). Values: mean ± standard deviation over the draws; draws where the ball fell are counted in 'Fallen' and left out of the other columns. Δ: mean of the paired differences over the same draws ± standard error, in **bold** when significant (paired t-test at the 5 % level: |Δ| > t × standard error, t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws).

The brain is the PID with its LQR gains (Kp = 1.89, Ki = 1.75, Kd = 0.80), its latency compensation, its integration zone and its aimed bounce (default settings): it is judged by the objectives and sections 2 to 9. Sections 0 and 10 validate the simulator: they use a frozen reference controller (hand-tuned gains Kp = 1, Ki = 1, ζ = 0.7, without latency compensation, error integrated everywhere, ball never considered in the air) and the motors of the first robot (single ramp).


## Objectives of the brain (full chain with the camera)

| Objective | Measured | Target | Met |
|---|---|---|---|
| Thrown ball caught (±10°) | 93 ± 1 % | ≥ 90 % of the physical limit | yes |
| Circle with 60 ms of latency | 0.87 ± 0.01 mm | < 2 mm | yes |
| Back to the center in less than 2 s | 4.06 ± 0.28 mm (calibration error: none) | ≤ 5 mm between 2 s and 4 s | yes |
| No fall in the robustness variants | 0/90 falls | 0 | yes |

Judged on the mean over the draws. Thrown ball: maximum speed caught at ±10° divided by the physical limit (section 2). Return: start 10 cm away, setpoint at the center, worst of the three calibration errors of the plate of section 4 (none, nominal 0.5°/-0.3°, ×2), since the actual one is unknown. Robustness: every variant and every draw of section 4.


## 0. From the simple model to the full chain (reference scenario, reference controller)

| Level | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Factor changed | Δ (effect of the factor) |
|---|---|---|---|---|---|---|
| Simple physics: angle-driven plate | 0/10 | 0.83 ± 0.01 s | 38.94 ± 0.83 mm | 29.90 ± 1.31 mm | - | - |
| Simple physics: 3-arm robot | 0/10 | 0.84 ± 0.01 s | 40.73 ± 0.71 mm | 30.56 ± 0.80 mm | mechanism | +0.66 ± 0.37 mm |
| MuJoCo: angle-driven plate | 0/10 | 0.84 ± 0.01 s | 40.72 ± 0.97 mm | 31.85 ± 1.19 mm | physics | **+1.95 ± 0.12 mm** |
| MuJoCo: 3-arm robot | 0/10 | 0.84 ± 0.01 s | 41.29 ± 0.65 mm | 31.62 ± 1.20 mm | physics | **+1.06 ± 0.17 mm** |
| MuJoCo: 3-arm robot, sensor without noise | 0/10 | 0.84 ± 0.00 s | 41.52 ± 0.12 mm | 32.53 ± 0.10 mm | noise | **+0.91 ± 0.38 mm** |
| MuJoCo: 3-arm robot + camera | 0/10 | 0.84 ± 0.00 s | 41.71 ± 0.13 mm | 32.45 ± 0.12 mm | camera | **-0.08 ± 0.02 mm** |

Return to the center (start 10 cm away, setpoint at the center until 4 s). Objective "back to the center in less than 2 s": max distance between 2 s and 4 s ≤ 5 mm. Angle-driven plate: its two angles follow the command with the profile of the motors (speed, ramps), without steps; the mechanism adds the arms (kinematics, microsteps: the motors turn about twice as much as the plate, so their ramps tilt it more slowly).

| Level | Circle error | Factor changed | Δ circle error (effect of the factor) | θx jitter | Airborne |
|---|---|---|---|---|---|
| Simple physics: angle-driven plate | 4.69 ± 0.50 mm | - | - | 0.59 ± 0.06° | - |
| Simple physics: 3-arm robot | 4.77 ± 0.38 mm | mechanism | +0.08 ± 0.18 mm | 0.59 ± 0.06° | - |
| MuJoCo: angle-driven plate | 5.23 ± 0.60 mm | physics | **+0.54 ± 0.07 mm** | 0.60 ± 0.06° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot | 5.33 ± 0.36 mm | physics | **+0.56 ± 0.06 mm** | 0.59 ± 0.06° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot, sensor without noise | 4.25 ± 0.01 mm | noise | **-1.08 ± 0.12 mm** | 0.07 ± 0.00° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot + camera | 4.58 ± 0.03 mm | camera | **+0.33 ± 0.01 mm** | 0.12 ± 0.00° | 0.0 ± 0.0 % |

Circle: 6 cm radius in 4 s, error = root-mean-square distance from 12 s to 20 s. Factor changed: compared with the level above that differs only by this factor (the mechanism: the simple plate; the physics: the same level in simple physics; the noise: the synthetic sensor of standard deviation 1 mm, made perfect; the camera: the perfect sensor). The synthetic sensor is about 5 times noisier than the camera: its noise, not the camera, makes the gap between them.


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
| ±10° | 0.66 ± 0.00 m/s | 0.71 m/s | 93 ± 1 % | yes |
| ±20° | 0.85 ± 0.01 m/s | 1.00 m/s | 85 ± 1 % | yes |
| ±10°, without waiting for the speed | 0.65 ± 0.02 m/s | 0.71 m/s | 91 ± 3 % | NO, exceptions at 0.65 m/s |
| ±10°, single motor ramp of 50 rad/s² (before B4) | 0.58 ± 0.01 m/s | 0.71 m/s | 81 ± 1 % | yes |

The first image arrives one latency (20 ms) after the throw; the brain waits for the second one, which gives the speed of the ball, before tilting the plate: at the first one, the position alone would tilt it the wrong way. Motor ramps: 400 rad/s² for the large moves (gap above 0.1 rad), 50 rad/s² for the small corrections.


## 3. Push on the ball at rest at the center (t = 1 s)

| Max tilt | Max impulse absorbed (Δv) | Physical limit √(2·c·g·sin θmax·d) | Share of the limit | Monotonic around the threshold (draw 0) |
|---|---|---|---|---|
| ±10° | 0.51 ± 0.01 m/s | 0.55 m/s | 93 ± 1 % | yes |

Maximum speeds found by bisection, to 0.01 m/s, between 0.1 and 1.5 m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).


## 4. Robustness to physical parameters and to a dropped ball (controller unchanged, tuned for the ping-pong ball)

| Variant | Fallen | Max distance between 2 s and 4 s | Circle error | Δ circle error vs nominal | Airborne |
|---|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | 0/10 | 1.48 ± 0.09 mm | 0.80 ± 0.01 mm | - | 4.1 ± 1.3 % |
| Solid ball (k = 2/5) | 0/10 | 1.95 ± 0.30 mm | 2.71 ± 0.01 mm | **+1.90 ± 0.01 mm** | 5.2 ± 1.2 % |
| Ball 10× heavier (27 g) | 0/10 | 1.40 ± 0.11 mm | 0.77 ± 0.01 mm | **-0.04 ± 0.01 mm** | 1.5 ± 0.5 % |
| Friction μ = 0.2 | 0/10 | 1.36 ± 0.08 mm | 0.77 ± 0.01 mm | **-0.03 ± 0.00 mm** | 4.6 ± 0.7 % |
| Friction μ = 0.02 | 0/10 | 2.08 ± 0.18 mm | 0.77 ± 0.01 mm | **-0.04 ± 0.01 mm** | 7.1 ± 1.1 % |
| Plate 2× heavier (0.5 kg) | 0/10 | 1.51 ± 0.14 mm | 0.79 ± 0.01 mm | **-0.01 ± 0.00 mm** | 4.3 ± 1.0 % |
| No calibration error (0°, 0°) | 0/10 | 4.06 ± 0.28 mm | 0.81 ± 0.01 mm | +0.00 ± 0.00 mm | 6.0 ± 1.4 % |
| Calibration error ×2 (1°, -0.6°) | 0/10 | 1.88 ± 0.14 mm | 0.79 ± 0.01 mm | **-0.01 ± 0.00 mm** | 4.4 ± 0.7 % |
| Ball dropped from 30.5 cm (ITTF height) | 0/10 | 14.28 ± 0.94 mm | 0.81 ± 0.01 mm | +0.00 ± 0.00 mm | 18.5 ± 0.6 % |

Reference scenario. Dropped ball: released without speed above the start point, 30.5 cm above the plate. A ball in the air beyond the edge of the tilted plate counts as fallen only once it is level with the plate.


## 5. Dropped ball: where and from how high (setpoint: center)

| Distance from the center | Height | Fallen, reacting in the air | Fallen, aimed bounce | Airborne | Max distance from the center between 4 s and 6 s | Height measured in the air: error (RMS) | On the plate but measured in the air |
|---|---|---|---|---|---|---|---|
| 0 cm | 10.0 cm | 0/10 | 0/10 | 34.5 ± 1.1 % | 0.25 ± 0.04 mm | 3.00 ± 0.20 mm | 0.0 ± 0.0 % |
| 0 cm | 20.0 cm | 0/10 | 0/10 | 45.4 ± 1.7 % | 0.44 ± 0.29 mm | 2.29 ± 0.17 mm | 0.0 ± 0.0 % |
| 0 cm | 30.5 cm | 0/10 | 0/10 | 58.8 ± 1.9 % | 0.62 ± 0.32 mm | 2.24 ± 0.12 mm | 0.0 ± 0.0 % |
| 5 cm | 10.0 cm | 0/10 | 0/10 | 33.5 ± 1.4 % | 0.32 ± 0.08 mm | 2.45 ± 0.36 mm | 0.0 ± 0.0 % |
| 5 cm | 20.0 cm | 10/10 | 0/10 | 47.1 ± 1.1 % | 0.70 ± 0.23 mm | 2.23 ± 0.21 mm | 0.0 ± 0.0 % |
| 5 cm | 30.5 cm | 10/10 | 0/10 | 58.9 ± 2.0 % | 1.21 ± 0.43 mm | 2.11 ± 0.13 mm | 0.0 ± 0.0 % |
| 10 cm | 10.0 cm | 0/10 | 0/10 | 32.9 ± 1.5 % | 0.42 ± 0.12 mm | 3.81 ± 0.37 mm | 0.0 ± 0.0 % |
| 10 cm | 20.0 cm | 10/10 | 0/10 | 45.0 ± 1.5 % | 0.70 ± 0.16 mm | 3.19 ± 0.26 mm | 0.0 ± 0.0 % |
| 10 cm | 30.5 cm | 10/10 | 0/10 | 55.0 ± 1.6 % | 1.15 ± 0.42 mm | 6.38 ± 0.18 mm | 0.0 ± 0.0 % |

Ball released without speed above a point of the line from the center to the start point of the reference scenario, 6 s per drop. Reacting in the air: the brain before B5, which drives the plate from the measured position of the ball also while it is in the air. Aimed bounce (the brain): in the air, the plate is tilted so that the next bounce sends the ball toward the setpoint (controller/flight.py, controller/plate.py); the other columns are those of the brain. Height: measured by the camera from the apparent size of the ball; in the air above 15 mm (true height), on the plate below 5 mm. A ball partly out of the image is not measured: the last measurement is kept.


## 6. Camera latency: without and with compensation by prediction

| Latency | Compensation | Fallen | Max distance between 2 s and 4 s | Δ (with - without) | Circle error | Δ circle error (with - without) |
|---|---|---|---|---|---|---|
| 20 ms | without | 0/10 | 1.84 ± 0.42 mm | - | 3.17 ± 0.01 mm | - |
| 20 ms | with | 0/10 | 1.48 ± 0.09 mm | **-0.36 ± 0.14 mm** | 0.80 ± 0.01 mm | **-2.36 ± 0.00 mm** |
| 60 ms | without | 0/10 | 3.48 ± 0.28 mm | - | 7.87 ± 0.03 mm | - |
| 60 ms | with | 0/10 | 1.26 ± 0.72 mm | **-2.22 ± 0.23 mm** | 0.87 ± 0.01 mm | **-7.00 ± 0.01 mm** |
| 100 ms | without | 0/10 | 19.88 ± 4.56 mm | - | 12.72 ± 0.04 mm | - |
| 100 ms | with | 0/10 | 5.34 ± 5.13 mm | **-14.55 ± 1.15 mm** | 1.14 ± 0.03 mm | **-11.58 ± 0.02 mm** |


## 7. Integral action: error integrated everywhere or only near the setpoint

| Error integrated | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Δ (zone - everywhere) | Circle error | Δ circle error (zone - everywhere) |
|---|---|---|---|---|---|---|---|
| everywhere | 0/10 | 0.74 ± 0.02 s | 25.61 ± 0.63 mm | 14.20 ± 0.40 mm | - | 0.80 ± 0.01 mm | - |
| within 20 mm | 0/10 | 1.19 ± 0.03 s | 4.77 ± 0.13 mm | 1.48 ± 0.09 mm | **-12.71 ± 0.13 mm** | 0.80 ± 0.01 mm | +0.00 ± 0.00 mm |

Reference scenario, latency 20 ms with compensation. With the zone, each axis integrates its error only while it is below 20 mm: the integral no longer builds up during the approach, and only corrects the calibration error of the plate.


## 8. Gains of the PID: hand-tuned or by LQR

| Gains | Kp, Ki, Kd | Fallen | Max distance between 2 s and 4 s (worst calibration error) | Circle error | Δ (LQR - hand-tuned) | Circle error, 60 ms | Δ (LQR - hand-tuned) | θx jitter | Thrown ball caught (±10°) |
|---|---|---|---|---|---|---|---|---|---|
| hand-tuned | 1.00, 1.00, 0.58 | 0/10 | 14.14 ± 0.24 mm (×2) | 1.47 ± 0.01 mm | - | 1.66 ± 0.02 mm | - | 0.09 ± 0.01° | 0.67 ± 0.00 m/s |
| LQR | 1.89, 1.75, 0.80 | 0/10 | 4.06 ± 0.28 mm (none) | 0.80 ± 0.01 mm | **-0.67 ± 0.01 mm** | 0.87 ± 0.01 mm | **-0.78 ± 0.01 mm** | 0.08 ± 0.03° | 0.66 ± 0.00 m/s |

Latency 20 ms with compensation, integration zone, unless stated. LQR gains (controller/lqr.py, Bryson's rule): a position error of 20 cm, an integral of the error over 0.5 s and a tilt of 10° cost as much as one another. The stiffer loop corrects the calibration error of the plate faster.


## 9. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms

| Compensation | Fallen | Error (2nd lap) | Δ (with - without) |
|---|---|---|---|
| without | 0/10 | 6.78 ± 0.03 mm | - |
| with | 0/10 | 6.04 ± 0.01 mm | **-0.74 ± 0.01 mm** |


## 10. Simple physics → MuJoCo: the gap, term by term (synthetic sensor, reference controller)

Angle-driven plate:

| Model | Fallen | Circle error | Δ vs basic model | Share of the MuJoCo gap | Max distance between 2 s and 4 s | Δ vs basic model | Share of the MuJoCo gap |
|---|---|---|---|---|---|---|---|
| Simple physics, basic model | 0/10 | 4.69 ± 0.50 mm | - | - | 29.90 ± 1.31 mm | - | - |
| + r·θ̈ (the plate turns under the ball) | 0/10 | 4.62 ± 0.52 mm | -0.07 ± 0.06 mm | -13 % | 29.70 ± 1.36 mm | -0.19 ± 0.19 mm | -10 % |
| + rotating-frame terms (Euler, centrifugal, Coriolis) | 0/10 | 4.83 ± 0.63 mm | +0.15 ± 0.07 mm | 27 % | 32.78 ± 1.29 mm | **+2.88 ± 0.13 mm** | 147 % |
| + exact projection of gravity | 0/10 | 4.68 ± 0.51 mm | -0.01 ± 0.02 mm | -1 % | 29.93 ± 1.29 mm | +0.03 ± 0.04 mm | 2 % |
| + servo lag (τ = kv/kp = 10 ms) | 0/10 | 5.17 ± 0.53 mm | **+0.48 ± 0.03 mm** | 89 % | 29.39 ± 1.30 mm | **-0.51 ± 0.03 mm** | -26 % |
| + all of the above | 0/10 | 5.13 ± 0.57 mm | **+0.44 ± 0.06 mm** | 82 % | 31.70 ± 1.16 mm | **+1.80 ± 0.11 mm** | 92 % |
| MuJoCo | 0/10 | 5.23 ± 0.60 mm | **+0.54 ± 0.07 mm** | 100 % | 31.85 ± 1.19 mm | **+1.95 ± 0.12 mm** | 100 % |

3-arm robot:

| Model | Fallen | Circle error | Δ vs basic model | Share of the MuJoCo gap | Max distance between 2 s and 4 s | Δ vs basic model | Share of the MuJoCo gap |
|---|---|---|---|---|---|---|---|
| Simple physics, basic model | 0/10 | 4.77 ± 0.38 mm | - | - | 30.56 ± 0.80 mm | - | - |
| + r·θ̈ (the plate turns under the ball) | 0/10 | 4.80 ± 0.42 mm | +0.03 ± 0.08 mm | 5 % | 30.50 ± 1.13 mm | -0.06 ± 0.19 mm | -6 % |
| + rotating-frame terms (Euler, centrifugal, Coriolis) | 0/10 | 4.88 ± 0.35 mm | +0.11 ± 0.06 mm | 19 % | 32.36 ± 0.88 mm | **+1.80 ± 0.15 mm** | 171 % |
| + exact projection of gravity | 0/10 | 4.83 ± 0.40 mm | +0.07 ± 0.06 mm | 12 % | 30.61 ± 0.95 mm | +0.05 ± 0.10 mm | 5 % |
| + servo lag (τ = kv/kp = 10 ms) | 0/10 | 5.35 ± 0.40 mm | **+0.58 ± 0.06 mm** | 103 % | 30.20 ± 0.86 mm | **-0.37 ± 0.09 mm** | -35 % |
| + all of the above | 0/10 | 5.38 ± 0.43 mm | **+0.61 ± 0.10 mm** | 109 % | 31.57 ± 1.10 mm | **+1.01 ± 0.16 mm** | 96 % |
| MuJoCo | 0/10 | 5.33 ± 0.36 mm | **+0.56 ± 0.06 mm** | 100 % | 31.62 ± 1.20 mm | **+1.06 ± 0.17 mm** | 100 % |

Each addition is switched on alone in the simple physics (simulation/simple_physics.py). A gap is explained when the share predicted by a model is within 80 to 120 %. The additions do not add up: on the return to the center, the rotating-frame terms and the servo lag act in opposite directions.

