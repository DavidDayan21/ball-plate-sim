# Measurement campaign report for the simulated balancing robot

Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.

Each case is simulated over 10 draw(s). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise seed k and a start position shifted by up to ±0.5 mm on each axis (the only source of variation for the camera cases). Values: mean ± standard deviation over the draws; draws where the ball fell are counted in 'Fallen' and left out of the other columns. Δ: mean of the paired differences over the same draws ± standard error, in **bold** when significant (paired t-test at the 5 % level: |Δ| > t × standard error, t = 97.5 % quantile of Student's distribution, 2.26 for 10 draws). Sections 0 and 10, which validate the simulator: 40 draws.

The brain is the PID with its LQR gains (Kp = 1.89, Ki = 1.75, Kd = 0.80), its latency compensation, its integration zone and its aimed bounce (default settings): it is judged by the objectives and sections 2 to 9. Sections 0 and 10 validate the simulator: they use a frozen reference controller (hand-tuned gains Kp = 1, Ki = 1, ζ = 0.7, without latency compensation, error integrated everywhere, ball never considered in the air) and the motors of the first robot (single ramp).


## Objectives of the brain (full chain with the camera)

| Objective | Measured | Target | Met |
|---|---|---|---|
| Thrown ball caught (±10°) | 93 ± 1 % | ≥ 90 % of the physical limit | yes |
| Circle with 60 ms of latency | 0.90 ± 0.02 mm | < 2 mm | yes |
| Back to the center in less than 2 s | 3.39 ± 0.03 mm (calibration error: none) | ≤ 5 mm between 2 s and 4 s | yes |
| No fall in the robustness variants | 0/90 falls | 0 | yes |

Judged on the mean over the draws. Thrown ball: maximum speed caught at ±10° divided by the physical limit (section 2). Return: start 10 cm away, setpoint at the center, worst of the three calibration errors of the plate of section 4 (none, nominal 0.5°/-0.3°, ×2), since the actual one is unknown. Robustness: every variant and every draw of section 4.


## 0. From the simple model to the full chain (reference scenario, reference controller)

| Level | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Factor changed | Δ (effect of the factor) |
|---|---|---|---|---|---|---|
| Simple physics: angle-driven plate | 0/40 | 0.83 ± 0.01 s | 38.89 ± 0.94 mm | 29.19 ± 1.29 mm | - | - |
| Simple physics: 3-arm robot | 0/40 | 0.84 ± 0.01 s | 40.18 ± 1.07 mm | 30.25 ± 1.14 mm | mechanism | **+1.06 ± 0.19 mm** |
| MuJoCo: angle-driven plate | 0/40 | 0.83 ± 0.01 s | 40.70 ± 1.03 mm | 31.00 ± 1.30 mm | physics | **+1.82 ± 0.13 mm** |
| MuJoCo: 3-arm robot | 0/40 | 0.83 ± 0.01 s | 41.12 ± 1.09 mm | 31.07 ± 1.31 mm | physics | **+0.83 ± 0.08 mm** |
| MuJoCo: 3-arm robot, sensor without noise | 0/40 | 0.84 ± 0.00 s | 41.49 ± 0.13 mm | 32.49 ± 0.10 mm | noise | **+1.42 ± 0.21 mm** |
| MuJoCo: 3-arm robot + camera | 0/40 | 0.84 ± 0.00 s | 41.68 ± 0.12 mm | 32.40 ± 0.11 mm | camera | **-0.09 ± 0.01 mm** |

Return to the center (start 10 cm away, setpoint at the center until 4 s). Objective "back to the center in less than 2 s": max distance between 2 s and 4 s ≤ 5 mm. Angle-driven plate: its two angles follow the command with the profile of the motors (speed, ramps), without steps; the mechanism adds the arms (kinematics, microsteps: the motors turn about twice as much as the plate, so their ramps tilt it more slowly).

| Level | Circle error | Factor changed | Δ circle error (effect of the factor) | θx jitter | Airborne |
|---|---|---|---|---|---|
| Simple physics: angle-driven plate | 4.70 ± 0.37 mm | - | - | 0.59 ± 0.06° | - |
| Simple physics: 3-arm robot | 4.81 ± 0.35 mm | mechanism | +0.11 ± 0.07 mm | 0.59 ± 0.06° | - |
| MuJoCo: angle-driven plate | 5.37 ± 0.44 mm | physics | **+0.67 ± 0.04 mm** | 0.60 ± 0.06° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot | 5.31 ± 0.40 mm | physics | **+0.50 ± 0.04 mm** | 0.59 ± 0.06° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot, sensor without noise | 4.27 ± 0.01 mm | noise | **-1.04 ± 0.06 mm** | 0.07 ± 0.00° | 0.0 ± 0.0 % |
| MuJoCo: 3-arm robot + camera | 4.60 ± 0.03 mm | camera | **+0.33 ± 0.01 mm** | 0.12 ± 0.00° | 0.0 ± 0.0 % |

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
| ±10° | 0.67 ± 0.00 m/s | 0.71 m/s | 93 ± 1 % | yes |
| ±20° | 0.85 ± 0.01 m/s | 1.00 m/s | 85 ± 1 % | yes |
| ±10°, without waiting for the speed | 0.59 ± 0.01 m/s | 0.71 m/s | 83 ± 1 % | NO, exceptions at 0.58 m/s |
| ±10°, single motor ramp of 50 rad/s² (before B4) | 0.57 ± 0.01 m/s | 0.71 m/s | 80 ± 1 % | yes |

The first image arrives one latency (20 ms) after the throw; the brain waits for the second one, which gives the speed of the ball, before tilting the plate: at the first one, the position alone would tilt it the wrong way. Motor ramps: 400 rad/s² for the large moves (gap above 0.1 rad) of a ball faster than 0.3 m/s, 50 rad/s² otherwise.


## 3. Push on the ball at rest at the center (t = 1 s)

| Max tilt | Max impulse absorbed (Δv) | Physical limit √(2·c·g·sin θmax·d) | Share of the limit | Monotonic around the threshold (draw 0) |
|---|---|---|---|---|
| ±10° | 0.51 ± 0.01 m/s | 0.55 m/s | 93 ± 1 % | yes |

Maximum speeds found by bisection, to 0.01 m/s, between 0.1 and 1.5 m/s. Distance d: from the start to the edge (throw), from the center to the edge (push).


## 4. Robustness to physical parameters and to a dropped ball (controller unchanged, tuned for the ping-pong ball)

| Variant | Fallen | Max distance between 2 s and 4 s | Circle error | Δ circle error vs nominal | Airborne |
|---|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | 0/10 | 1.28 ± 0.05 mm | 0.78 ± 0.01 mm | - | 1.4 ± 0.7 % |
| Solid ball (k = 2/5) | 0/10 | 2.74 ± 0.08 mm | 0.68 ± 0.01 mm | **-0.10 ± 0.00 mm** | 1.8 ± 0.5 % |
| Ball 10× heavier (27 g) | 0/10 | 1.27 ± 0.05 mm | 0.78 ± 0.01 mm | +0.00 ± 0.00 mm | 0.3 ± 0.0 % |
| Friction μ = 0.2 | 0/10 | 1.27 ± 0.04 mm | 0.77 ± 0.01 mm | -0.00 ± 0.00 mm | 1.1 ± 0.5 % |
| Friction μ = 0.02 | 0/10 | 3.14 ± 0.08 mm | 0.77 ± 0.01 mm | -0.00 ± 0.00 mm | 1.9 ± 0.4 % |
| Plate 2× heavier (0.5 kg) | 0/10 | 1.26 ± 0.04 mm | 0.78 ± 0.01 mm | -0.00 ± 0.00 mm | 0.0 ± 0.0 % |
| No calibration error (0°, 0°) | 0/10 | 3.39 ± 0.03 mm | 0.78 ± 0.01 mm | +0.00 ± 0.01 mm | 0.4 ± 0.6 % |
| Calibration error ×2 (1°, -0.6°) | 0/10 | 1.80 ± 0.02 mm | 0.78 ± 0.01 mm | -0.00 ± 0.00 mm | 2.0 ± 0.7 % |
| Ball dropped from 30.5 cm (ITTF height) | 0/10 | 13.67 ± 1.83 mm | 0.78 ± 0.01 mm | +0.00 ± 0.00 mm | 17.5 ± 0.5 % |

Reference scenario. Dropped ball: released without speed above the start point, 30.5 cm above the plate. A ball in the air beyond the edge of the tilted plate counts as fallen only once it is level with the plate.


## 5. Dropped ball: where and from how high (setpoint: center)

| Distance from the center | Height | Fallen, reacting in the air | Fallen, aimed bounce | Airborne | Max distance from the center between 4 s and 6 s | Height measured in the air: error (RMS) | On the plate but measured in the air |
|---|---|---|---|---|---|---|---|
| 0 cm | 10.0 cm | 0/10 | 0/10 | 34.5 ± 1.7 % | 0.29 ± 0.03 mm | 3.10 ± 0.20 mm | 0.0 ± 0.0 % |
| 0 cm | 20.0 cm | 0/10 | 0/10 | 46.0 ± 1.4 % | 0.39 ± 0.12 mm | 2.23 ± 0.22 mm | 0.0 ± 0.0 % |
| 0 cm | 30.5 cm | 0/10 | 0/10 | 58.3 ± 1.5 % | 0.78 ± 0.24 mm | 2.18 ± 0.17 mm | 0.0 ± 0.0 % |
| 5 cm | 10.0 cm | 0/10 | 0/10 | 32.8 ± 1.9 % | 0.28 ± 0.08 mm | 2.55 ± 0.23 mm | 0.0 ± 0.0 % |
| 5 cm | 20.0 cm | 0/10 | 0/10 | 46.9 ± 2.0 % | 0.60 ± 0.31 mm | 2.23 ± 0.14 mm | 0.0 ± 0.0 % |
| 5 cm | 30.5 cm | 10/10 | 0/10 | 59.3 ± 1.8 % | 1.25 ± 0.30 mm | 2.05 ± 0.12 mm | 0.0 ± 0.0 % |
| 10 cm | 10.0 cm | 0/10 | 0/10 | 33.8 ± 1.5 % | 0.47 ± 0.12 mm | 4.11 ± 0.67 mm | 0.0 ± 0.0 % |
| 10 cm | 20.0 cm | 10/10 | 0/10 | 44.0 ± 1.5 % | 0.76 ± 0.23 mm | 3.13 ± 0.18 mm | 0.0 ± 0.0 % |
| 10 cm | 30.5 cm | 10/10 | 0/10 | 54.2 ± 1.8 % | 1.01 ± 0.26 mm | 6.43 ± 0.18 mm | 0.0 ± 0.0 % |

Ball released without speed above a point of the line from the center to the start point of the reference scenario, 6 s per drop. Reacting in the air: the brain before B5, which drives the plate from the measured position of the ball also while it is in the air. Aimed bounce (the brain): in the air, the plate is tilted so that the next bounce sends the ball toward the setpoint (controller/flight.py, controller/plate.py); the other columns are those of the brain. Height: measured by the camera from the apparent size of the ball; in the air above 15 mm (true height), on the plate below 5 mm. A ball partly out of the image is not measured: the last measurement is kept.


## 6. Camera latency: without and with compensation by prediction

| Latency | Compensation | Fallen | Max distance between 2 s and 4 s | Δ (with - without) | Circle error | Δ circle error (with - without) |
|---|---|---|---|---|---|---|
| 20 ms | without | 0/10 | 1.38 ± 0.08 mm | - | 3.16 ± 0.02 mm | - |
| 20 ms | with | 0/10 | 1.28 ± 0.05 mm | **-0.11 ± 0.03 mm** | 0.78 ± 0.01 mm | **-2.39 ± 0.01 mm** |
| 60 ms | without | 0/10 | 1.49 ± 0.06 mm | - | 7.90 ± 0.02 mm | - |
| 60 ms | with | 0/10 | 1.36 ± 0.05 mm | **-0.13 ± 0.02 mm** | 0.90 ± 0.02 mm | **-7.01 ± 0.01 mm** |
| 100 ms | without | 0/10 | 35.33 ± 0.46 mm | - | 12.85 ± 0.05 mm | - |
| 100 ms | with | 0/10 | 1.06 ± 0.07 mm | **-34.28 ± 0.14 mm** | 1.17 ± 0.03 mm | **-11.68 ± 0.02 mm** |

| Latency | Fast ramps of the motors | Fallen | Max distance between 2 s and 4 s | Δ (brain - always) | Circle error | Airborne |
|---|---|---|---|---|---|---|
| 20 ms | whatever the speed of the ball | 0/10 | 1.44 ± 0.11 mm | - | 0.78 ± 0.01 mm | 4.3 ± 1.0 % |
| 20 ms | only above 0.3 m/s (the brain) | 0/10 | 1.28 ± 0.05 mm | **-0.16 ± 0.04 mm** | 0.78 ± 0.01 mm | 1.4 ± 0.7 % |
| 100 ms | whatever the speed of the ball | 0/10 | 3.53 ± 2.12 mm | - | 1.17 ± 0.02 mm | 6.9 ± 1.7 % |
| 100 ms | only above 0.3 m/s (the brain) | 0/10 | 1.06 ± 0.07 mm | **-2.47 ± 0.68 mm** | 1.17 ± 0.03 mm | 0.0 ± 0.0 % |

With compensation. The fast ramp of the motors throws a slow ball off the plate: 10 cm from the center, the plate under the ball accelerates downward faster than g beyond about 100 rad/s² of plate (r·θ̈ > g), about 200 rad/s² of motor. With a large latency, the ball then keeps bouncing; the brain keeps the fast ramps for a fast ball (thrown, pushed), which needs them.


## 7. Integral action: error integrated everywhere or only near the setpoint

| Error integrated | Fallen | Arrival (first time < 5 mm) | Overshoot after arrival | Max distance between 2 s and 4 s | Δ (zone - everywhere) | Circle error | Δ circle error (zone - everywhere) |
|---|---|---|---|---|---|---|---|
| everywhere | 0/10 | 1.51 ± 0.99 s | 18.22 ± 11.48 mm | 14.74 ± 0.09 mm | - | 0.77 ± 0.01 mm | - |
| within 20 mm | 0/10 | 1.16 ± 0.01 s | 4.88 ± 0.09 mm | 1.28 ± 0.05 mm | **-13.47 ± 0.03 mm** | 0.78 ± 0.01 mm | +0.01 ± 0.00 mm |

Reference scenario, latency 20 ms with compensation. With the zone, each axis integrates its error only while it is below 20 mm: the integral no longer builds up during the approach, and only corrects the calibration error of the plate. What it still builds up over the last 20 mm of the approach (a few tenths of a degree) is about what the nominal calibration error requires; without calibration error, it is in excess: the ball overshoots the center and comes back as the integral empties, which makes this case the worst return to the center (section 4).


## 8. Gains of the PID: hand-tuned or by LQR

| Gains | Kp, Ki, Kd | Fallen | Max distance between 2 s and 4 s (worst calibration error) | Circle error | Δ (LQR - hand-tuned) | Circle error, 60 ms | Δ (LQR - hand-tuned) | θx jitter | Thrown ball caught (±10°) |
|---|---|---|---|---|---|---|---|---|---|
| hand-tuned | 1.00, 1.00, 0.58 | 0/10 | 13.75 ± 0.04 mm (×2) | 1.47 ± 0.02 mm | - | 1.66 ± 0.02 mm | - | 0.08 ± 0.01° | 0.66 ± 0.01 m/s |
| LQR | 1.89, 1.75, 0.80 | 0/10 | 3.39 ± 0.03 mm (none) | 0.78 ± 0.01 mm | **-0.69 ± 0.01 mm** | 0.90 ± 0.02 mm | **-0.76 ± 0.01 mm** | 0.10 ± 0.02° | 0.67 ± 0.00 m/s |

Latency 20 ms with compensation, integration zone, unless stated. LQR gains (controller/lqr.py, Bryson's rule): a position error of 20 cm, an integral of the error over 0.5 s and a tilt of 10° cost as much as one another. The stiffer loop corrects the calibration error of the plate faster.


## 9. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms

| Compensation | Fallen | Error (2nd lap) | Δ (with - without) |
|---|---|---|---|
| without | 0/10 | 6.77 ± 0.04 mm | - |
| with | 0/10 | 6.05 ± 0.01 mm | **-0.73 ± 0.01 mm** |


## 10. Simple physics → MuJoCo: the gap, term by term (synthetic sensor, reference controller)

Angle-driven plate:

| Model | Fallen | Circle error | Δ vs basic model | Share of the MuJoCo gap | Max distance between 2 s and 4 s | Δ vs basic model | Share of the MuJoCo gap |
|---|---|---|---|---|---|---|---|
| Simple physics, basic model | 0/40 | 4.70 ± 0.37 mm | - | - | 29.19 ± 1.29 mm | - | - |
| + r·θ̈ (the plate turns under the ball) | 0/40 | 4.69 ± 0.41 mm | -0.01 ± 0.03 mm | -1 % | 28.89 ± 1.31 mm | **-0.30 ± 0.09 mm** | -16 % |
| + rotating-frame terms (Euler, centrifugal, Coriolis) | 0/40 | 4.96 ± 0.49 mm | **+0.26 ± 0.04 mm** | 40 % | 32.17 ± 1.26 mm | **+2.98 ± 0.07 mm** | 164 % |
| + exact projection of gravity | 0/40 | 4.70 ± 0.37 mm | -0.00 ± 0.01 mm | -0 % | 29.19 ± 1.30 mm | +0.01 ± 0.01 mm | 0 % |
| + servo lag (τ = kv/kp = 10 ms) | 0/40 | 5.21 ± 0.39 mm | **+0.51 ± 0.01 mm** | 76 % | 28.66 ± 1.33 mm | **-0.53 ± 0.03 mm** | -29 % |
| + all of the above | 0/40 | 5.28 ± 0.42 mm | **+0.58 ± 0.04 mm** | 86 % | 30.78 ± 1.38 mm | **+1.59 ± 0.12 mm** | 88 % |
| MuJoCo | 0/40 | 5.37 ± 0.44 mm | **+0.67 ± 0.04 mm** | 100 % | 31.00 ± 1.30 mm | **+1.82 ± 0.13 mm** | 100 % |

3-arm robot:

| Model | Fallen | Circle error | Δ vs basic model | Share of the MuJoCo gap | Max distance between 2 s and 4 s | Δ vs basic model | Share of the MuJoCo gap |
|---|---|---|---|---|---|---|---|
| Simple physics, basic model | 0/40 | 4.81 ± 0.35 mm | - | - | 30.25 ± 1.14 mm | - | - |
| + r·θ̈ (the plate turns under the ball) | 0/40 | 4.82 ± 0.40 mm | +0.02 ± 0.03 mm | 3 % | 30.10 ± 1.29 mm | -0.14 ± 0.08 mm | -17 % |
| + rotating-frame terms (Euler, centrifugal, Coriolis) | 0/40 | 4.75 ± 0.42 mm | -0.06 ± 0.03 mm | -12 % | 32.00 ± 1.17 mm | **+1.76 ± 0.06 mm** | 212 % |
| + exact projection of gravity | 0/40 | 4.75 ± 0.39 mm | -0.06 ± 0.03 mm | -11 % | 30.31 ± 1.25 mm | +0.07 ± 0.06 mm | 8 % |
| + servo lag (τ = kv/kp = 10 ms) | 0/40 | 5.32 ± 0.37 mm | **+0.52 ± 0.03 mm** | 103 % | 29.80 ± 1.28 mm | **-0.44 ± 0.05 mm** | -53 % |
| + all of the above | 0/40 | 5.35 ± 0.43 mm | **+0.54 ± 0.04 mm** | 108 % | 31.18 ± 1.26 mm | **+0.94 ± 0.08 mm** | 113 % |
| MuJoCo | 0/40 | 5.31 ± 0.40 mm | **+0.50 ± 0.04 mm** | 100 % | 31.07 ± 1.31 mm | **+0.83 ± 0.08 mm** | 100 % |

Each addition is switched on alone in the simple physics (simulation/simple_physics.py). A gap is explained when the share predicted by a model is within 80 to 120 %. The additions do not add up: on the return to the center, the rotating-frame terms and the servo lag act in opposite directions.


## 11. Adaptive control: the model gain c·g estimated online (latency compensation, camera)

| Ball, latency | Model gain | Fallen | g/(1 + k) | c·g used at 20 s | Circle error | Δ (estimated - fixed) | Max distance between 2 s and 4 s | Δ (estimated - fixed) |
|---|---|---|---|---|---|---|---|---|
| hollow (k = 2/3), 20 ms | fixed | 0/10 | 5.89 | 5.89 ± 0.00 | 0.81 ± 0.01 mm | - | 1.28 ± 0.05 mm | - |
| hollow (k = 2/3), 20 ms | estimated | 0/10 | 5.89 | 5.92 ± 0.00 | 0.78 ± 0.01 mm | **-0.04 ± 0.00 mm** | 1.28 ± 0.05 mm | +0.00 ± 0.00 mm |
| hollow (k = 2/3), 60 ms | fixed | 0/10 | 5.89 | 5.89 ± 0.00 | 0.89 ± 0.01 mm | - | 1.36 ± 0.05 mm | - |
| hollow (k = 2/3), 60 ms | estimated | 0/10 | 5.89 | 5.90 ± 0.01 | 0.90 ± 0.02 mm | +0.00 ± 0.00 mm | 1.36 ± 0.05 mm | +0.00 ± 0.00 mm |
| solid (k = 2/5), 20 ms | fixed | 0/10 | 7.01 | 5.89 ± 0.00 | 2.71 ± 0.02 mm | - | 2.74 ± 0.08 mm | - |
| solid (k = 2/5), 20 ms | estimated | 0/10 | 7.01 | 7.04 ± 0.00 | 0.68 ± 0.01 mm | **-2.04 ± 0.01 mm** | 2.74 ± 0.08 mm | +0.00 ± 0.00 mm |

c·g in (m/s²)/rad; fixed: the value of the hollow ball (nominal). The controller estimates c·g from the measured accelerations of the ball and the commands it sent (controller/adaptation.py): least squares, recursive, on low-pass filtered signals, from the nominal value at every attempt. It uses them only while the ball rolls (at least 3 cm/s) within 1 cm of its setpoint for at least 1 s: elsewhere the linear model does not hold (near rest; during a transient, the plate lags its command). In the reference scenario, only the circle (from 8 s) excites it: the return to the center (before 4 s) does not depend on it. The estimate replaces the nominal c·g in the feedforward and in the latency predictor; the gains of the PID stay those of the nominal model.


## 12. The remaining gaps of more than 0.5 mm, reproduced by our equations or traced to their cause

Our equations: the simple physics of the 3-arm robot with the full equation of section 10 and sliding (Coulomb friction μ: the ball slides when rolling would need more). A gap is explained when a model reproduces it within 80 to 120 %, as in section 10.

| Robustness variant (section 4): gap of the return to the center to the nominal case | MuJoCo, sensor without noise | Our equations, sensor without noise | Share reproduced | MuJoCo + camera (section 4) | Effect of the camera |
|---|---|---|---|---|---|
| Solid ball (k = 2/5) | **+1.13 ± 0.03 mm** | **+1.11 ± 0.03 mm** | 98 % | **+1.46 ± 0.03 mm** | +0.33 mm |
| No calibration error (0°, 0°) | **+2.08 ± 0.03 mm** | **+2.08 ± 0.03 mm** | 100 % | **+2.11 ± 0.02 mm** | +0.03 mm |
| Calibration error ×2 (1°, -0.6°) | **+0.92 ± 0.02 mm** | **+0.85 ± 0.03 mm** | 92 % | **+0.52 ± 0.01 mm** | -0.40 mm |
| Friction μ = 0.02 | **+1.80 ± 0.03 mm** | **+1.73 ± 0.03 mm** | 96 % | **+1.86 ± 0.02 mm** | +0.07 mm |
| Friction μ = 0.02, MuJoCo with its own friction cone (pyramidal, used until phase B) | **+2.69 ± 0.02 mm** | **+1.73 ± 0.03 mm** | 64 % | - | - |

The brain in both. The return to the center is the tail of the approach: what the integral has built up over the last 2 cm (section 7), against what the calibration error of the plate needs. A solid ball (c·g 19 % higher than the gains of the PID assume) and a plate without calibration error leave the integral in excess; a calibration error ×2 leaves it short, and the ball arrives later. With μ = 0.02, the ball slides while the plate brakes it (rolling at 10° needs μ ≥ 0.071, section 1): its path to the center changes, and the integral of one axis builds up more over the last 2 cm. The simulator uses Coulomb's round friction cone; MuJoCo's own default, a pyramid, allows up to 29 % less friction than μ·N between its axes (1 - 1/√2): with it, the ball slides more and the gap is larger (last row). Last column: the effect of the camera on each gap (MuJoCo + camera minus MuJoCo without noise).

| Effect of the noise of the synthetic sensor (section 0): noise removed | MuJoCo | Our equations | Share reproduced |
|---|---|---|---|
| Max distance between 2 s and 4 s | **+1.42 ± 0.21 mm** | **+1.30 ± 0.20 mm** | 92 % |
| Circle error | **-1.04 ± 0.06 mm** | **-1.09 ± 0.07 mm** | 105 % |

Reference controller of section 0, over 4 times more draws: the noise makes the return vary by more than 1 mm from one draw to the next. Our equations reproduce the effect: it does not come from MuJoCo. The motors cannot follow the noisy command with their gentle ramp, and lag it.

| Effect of the mechanism on the return (section 0), sensor without noise | Δ vs angle-driven plate | Share explained by the ramps |
|---|---|---|
| 3-arm robot | **+1.29 ± 0.01 mm** | - |
| 3-arm robot, ramps and speed of its motors ×2 | **+0.15 ± 0.01 mm** | 88 % |

Simple physics, reference controller, without noise to measure the effect precisely (with the noise of section 0: **+1.06 ± 0.19 mm**). The motors turn about twice as much as the plate (the lever of the arms, section 0): with the same ramps, the plate of the robot tilts more slowly. Motors twice as fast remove most of the gap; the rest comes from the kinematics.

| Latency 100 ms without compensation | Max distance between 2 s and 4 s | Share | Circle error | Share |
|---|---|---|---|---|
| MuJoCo + camera (section 6) | 35.33 ± 0.46 mm | - | 12.85 ± 0.05 mm | - |
| Our equations | 38.81 ± 0.59 mm | 110 % | 13.29 ± 0.11 mm | 103 % |
| Basic model of the simple physics | 12.11 ± 0.44 mm | 34 % | 12.47 ± 0.05 mm | 97 % |

Synthetic sensor with the mean error of the camera (σ = 0.2 mm). Without compensation, 100 ms of latency leaves the loop lightly damped: the ball oscillates around the center for seconds. The plate turning under the ball (r·θ̈) and the lag of the servos damp it even less: the full equation reproduces MuJoCo, the basic model does not.

Other gaps, traced to their cause:

- Error integrated everywhere (section 7): the spread of the arrival and of the overshoot is that of a threshold. On its first pass, the ball comes within 4.84 ± 0.28 mm of the center, below the 5 mm of the arrival in 6 draws out of 10; the arrival is then this first pass, otherwise the next one, more than a second later. The return between 2 s and 4 s does not depend on a threshold.

- Thrown ball without waiting for its speed (section 2), at 0.55 m/s: at the first image, the position alone tilts the plate toward the edge; it brakes the ball (tilt beyond -9.5°) at 0.18 ± 0.00 s instead of 0.12 ± 0.01 s, when the ball is at x = 2.07 ± 0.23 mm instead of -30.76 ± 3.99 mm. Near the threshold, the ball stops at the edge: on draw 0, the fastest throw caught stops 3.7 mm from it. The ball also hops a little while the plate brakes it (in the air 7.2 ± 0.4 % of the time at 0.55 m/s): a millimetre either way decides, hence the exceptions to monotonicity.

- Ball dropped from 30.5 cm (section 4): the max distance between 2 s and 4 s is measured while it still bounces (last flight above 15 mm at 2.53 ± 0.12 s). For this variant, the criterion is that the ball does not fall.


## 13. The AI brain (learned policy "imitation") next to the PID brain (full chain with the camera)

| Objective | PID brain | Met | AI brain | Met | Target |
|---|---|---|---|---|---|
| Thrown ball caught (±10°) | 93 ± 1 % | yes | 93 ± 1 % | yes | ≥ 90 % of the physical limit |
| Circle with 60 ms of latency | 0.90 ± 0.02 mm | yes | 1.15 ± 0.02 mm | yes | < 2 mm |
| Back to the center in less than 2 s | 3.39 ± 0.03 mm (calibration error: none) | yes | 2.65 ± 0.03 mm (calibration error: none) | yes | ≤ 5 mm between 2 s and 4 s |
| No fall in the robustness variants | 0/90 falls | yes | 0/90 falls | yes | 0 |

| Case | Fallen (PID) | PID brain | Fallen (AI) | AI brain | Δ AI - PID |
|---|---|---|---|---|---|
| Thrown ball (±10°), share of the physical limit | - | 93 ± 1 % | - | 93 ± 1 % | -0.6 ± 0.3 % |
| Push at the center, share of the physical limit | - | 93 ± 1 % | - | 94 ± 1 % | **+0.7 ± 0.3 %** |
| Nominal: return | 0/10 | 1.28 ± 0.05 mm | 0/10 | 0.78 ± 0.03 mm | **-0.50 ± 0.02 mm** |
| Nominal: circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.10 ± 0.01 mm | **+0.33 ± 0.00 mm** |
| Latency 60 ms: return | 0/10 | 1.36 ± 0.05 mm | 0/10 | 0.78 ± 0.05 mm | **-0.58 ± 0.02 mm** |
| Latency 60 ms: circle | 0/10 | 0.90 ± 0.02 mm | 0/10 | 1.15 ± 0.02 mm | **+0.25 ± 0.01 mm** |
| Latency 100 ms: return | 0/10 | 1.06 ± 0.07 mm | 0/10 | 0.90 ± 0.09 mm | **-0.15 ± 0.05 mm** |
| Latency 100 ms: circle | 0/10 | 1.17 ± 0.03 mm | 0/10 | 1.34 ± 0.02 mm | **+0.17 ± 0.01 mm** |
| Solid ball (k = 2/5): return | 0/10 | 2.74 ± 0.08 mm | 0/10 | 1.07 ± 0.06 mm | **-1.67 ± 0.03 mm** |
| Solid ball (k = 2/5): circle | 0/10 | 0.68 ± 0.01 mm | 0/10 | 1.49 ± 0.01 mm | **+0.81 ± 0.00 mm** |
| Ball 10× heavier (27 g): return | 0/10 | 1.27 ± 0.05 mm | 0/10 | 0.78 ± 0.05 mm | **-0.49 ± 0.02 mm** |
| Ball 10× heavier (27 g): circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.15 ± 0.01 mm | **+0.37 ± 0.00 mm** |
| Friction μ = 0.2: return | 0/10 | 1.27 ± 0.04 mm | 0/10 | 0.78 ± 0.03 mm | **-0.49 ± 0.02 mm** |
| Friction μ = 0.2: circle | 0/10 | 0.77 ± 0.01 mm | 0/10 | 1.10 ± 0.01 mm | **+0.33 ± 0.00 mm** |
| Friction μ = 0.02: return | 0/10 | 3.14 ± 0.08 mm | 0/10 | 1.93 ± 0.02 mm | **-1.21 ± 0.03 mm** |
| Friction μ = 0.02: circle | 0/10 | 0.77 ± 0.01 mm | 0/10 | 1.12 ± 0.01 mm | **+0.35 ± 0.00 mm** |
| Plate 2× heavier (0.5 kg): return | 0/10 | 1.26 ± 0.04 mm | 0/10 | 0.68 ± 0.07 mm | **-0.58 ± 0.03 mm** |
| Plate 2× heavier (0.5 kg): circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.12 ± 0.02 mm | **+0.34 ± 0.00 mm** |
| No calibration error (0°, 0°): return | 0/10 | 3.39 ± 0.03 mm | 0/10 | 2.65 ± 0.03 mm | **-0.73 ± 0.01 mm** |
| No calibration error (0°, 0°): circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.17 ± 0.01 mm | **+0.39 ± 0.00 mm** |
| Calibration error ×2 (1°, -0.6°): return | 0/10 | 1.80 ± 0.02 mm | 0/10 | 1.77 ± 0.04 mm | **-0.03 ± 0.01 mm** |
| Calibration error ×2 (1°, -0.6°): circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.46 ± 0.02 mm | **+0.68 ± 0.01 mm** |
| Ball dropped from 30.5 cm (ITTF height): return | 0/10 | 13.67 ± 1.83 mm | 0/10 | 16.99 ± 3.72 mm | **+3.32 ± 1.05 mm** |
| Ball dropped from 30.5 cm (ITTF height): circle | 0/10 | 0.78 ± 0.01 mm | 0/10 | 1.14 ± 0.02 mm | **+0.37 ± 0.01 mm** |

Same cases, same draws, same rules for both brains (sensor, motors, maximum tilt). Return: max distance between 2 s and 4 s; circle: error from 12 s to 20 s. The AI brain is a neural network of 2 layers of 128 neurons that drives alone; it learned by imitating the PID brain (training/imitate.py), on the synthetic sensor, never with the camera; it has no estimate of the flight of its own: it learned from the PID how to tilt the plate under a ball in the air. Command at rest at the center (standard deviation of θx from 2.5 s to 4 s), 60 ms of latency: 0.159 ± 0.040° for the PID, 0.129 ± 0.031° for the AI.

