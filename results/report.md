# Measurement campaign report for the simulated balancing robot

Simulated chain: MuJoCo (3-arm robot, motor ramps) + 160×120 camera + OpenCV + PID.


## 0. From the simple model to the full chain (reference scenario, same settings)

| Level | Fallen | Settling time (5 %) | Circle error | θx jitter | Airborne |
|---|---|---|---|---|---|
| Simple physics: angle-driven plate | no | 2.76 s | 3.34 mm | 0.55° | - |
| Simple physics: 3-arm robot | no | 2.72 s | 4.35 mm | 0.57° | - |
| MuJoCo: angle-driven plate | no | 3.98 s | 3.01 mm | 0.55° | 75.2 % |
| MuJoCo: 3-arm robot | no | 2.68 s | 5.46 mm | 0.58° | 0.0 % |
| MuJoCo: 3-arm robot + camera | no | 2.78 s | 4.61 mm | 0.12° | 0.0 % |


## 1. Theoretical limits and contact calibration

| Quantity | Theory | MuJoCo |
|---|---|---|
| Minimum friction to roll at 10° | 0.071 | - |
| Acceleration at 10°, friction 1.3×min (rolls) | 1.022 m/s² | 1.022 m/s² |
| Acceleration at 10°, friction 0.7×min (slides) | 1.227 m/s² | 1.234 m/s² |
| Restitution, drop from 30.5 cm onto a fixed plane (ITTF rule) | 0.85 to 0.91 | 0.874 |


## 2. Ball thrown from x = -10 cm toward the opposite edge (setpoint: center)

| Max tilt | Max speed caught | Physical limit √(2·c·g·sin θmax·d) |
|---|---|---|
| ±10° | 0.5 m/s | 0.71 m/s |
| ±20° | 0.6 m/s | 1.00 m/s |


## 3. Push on the ball at rest at the center (t = 1 s)

| Max impulse absorbed (Δv) | Physical limit (d = 15 cm, ±10°) |
|---|---|
| 0.4 m/s | 0.55 m/s |


## 4. Robustness to physical parameters (controller unchanged, tuned for the ping-pong ball)

| Variant | Fallen | Settling time (5 %) | Circle error | Airborne |
|---|---|---|---|---|
| Nominal (ping-pong, μ = 1) | no | 2.78 s | 4.61 mm | 0.0 % |
| Solid ball (k = 2/5) | no | 2.90 s | 7.11 mm | 0.0 % |
| Ball 10× heavier (27 g) | no | 2.78 s | 4.61 mm | 0.0 % |
| Friction μ = 0.2 | no | 2.78 s | 4.45 mm | 0.0 % |
| Friction μ = 0.02 | no | 2.84 s | 4.47 mm | 0.0 % |
| Plate 2× heavier (0.5 kg) | no | 2.78 s | 4.56 mm | 0.0 % |


## 5. Camera latency: without and with compensation by prediction

| Latency | Compensation | Fallen | Settling time (5 %) | Circle error |
|---|---|---|---|---|
| 20 ms | without | no | 2.78 s | 4.61 mm |
| 20 ms | with | no | 2.82 s | 1.47 mm |
| 60 ms | without | no | 2.68 s | 10.56 mm |
| 60 ms | with | no | 2.82 s | 1.66 mm |
| 100 ms | without | no | 2.62 s | 16.66 mm |
| 100 ms | with | no | 3.72 s | 1.94 mm |


## 6. Square trajectory (10 cm sides, 5 cm/s), latency 20 ms

| Compensation | Fallen | Error (2nd lap) |
|---|---|---|
| without | no | 9.81 mm |
| with | no | 9.34 mm |

