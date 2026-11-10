# ball-plate-sim

Simulation of a balancing robot ("ball and plate"): a plate tilted by 3 motors keeps
a ball balanced, catches it when it is thrown, and makes it follow a trajectory.
Goal: a playful piece of software, usable without any knowledge of code, to play with the robot and
compare two "brains" that drive it: a PID and a local AI. The physics is computed by MuJoCo.
Building the real robot remains a distant goal: the code will then be ready.

## Progress

| Stage | Content | Status |
|---|---|---|
| Models | Simple physics (derived equations) and MuJoCo (3D, contacts, closed chain) | Done |
| Control | Filtered PID, anti-windup, feedforward, latency compensation by prediction | Done |
| Mechanism | 3 arms, exact inverse and forward kinematics, stepper motors with ramps | Done |
| Sensor | Virtual camera, OpenCV detection, pinhole projection, height of the ball from its apparent size | Done |
| Robustness | Thrown ball, pushes, dropped ball, physical parameters, latency (`campaign.py`) | Done |
| Realistic contact | Ping-pong ball bounce compliant with the ITTF rule (e = 0.874) | Done |
| 3D sandbox | Throws, drops, bounces, pushes, setpoints, data of every attempt | Done |
| A. Reliable measurements | Bit-exact regression check, spread over N draws, robust metrics, simple physics → MuJoCo gap | Done |
| B. PID brain | Objectives in the report, integral limited to the neighborhood of the setpoint, aimed bounce of a dropped ball, thrown ball (start with the speed, fast motor ramps for the large moves), gains tuned by LQR, adaptive control | **In progress** |
| C. Local AI brain | Training in MuJoCo, same rules as the PID (sensor, motors, ±10°) | To do |
| D. Software for the general public | MuJoCo physics in our interface, choice of brain, comparison | To do |
| E. Construction (distant) | Design, real-world defects, complete error budget | Later |

Quantified goals for both brains: thrown ball caught up to at least 90 % of the physical limit,
error on the circle < 2 mm with 60 ms of latency, return to the center < 2 s (from 10 cm away, the ball
stays within 5 mm of the center from 2 s on, whatever the calibration error of the plate from 0 to 1°),
no fall in the robustness variants.

## Structure

```text
controller/              What would run on the real robot (independent of the simulator)
  pid.py                 PID: derivative filter, saturation, anti-windup, integration zone, feedforward
  plate.py               2 PIDs (x, y axes) + feedforward + latency predictor + aimed bounce
  flight.py              Ball in the air: ballistic tracking, prediction of the impact
  kinematics.py          Geometry, inverse (exact) and forward (Newton) kinematics
simulation/
  config.py              Single configuration, physical parameters, setpoint trajectories
  loop.py                Single closed loop: sensor -> controller -> motors -> physics
  simple_physics.py      Simple physics (theoretical reference)
  mujoco_physics.py      MuJoCo physics: generation of the 3D models, contact, interface, validation
  motor.py               Stepper motor: max speed, microsteps, limits, ramps
  sensor.py              Synthetic sensor: noise + latency
  camera.py              Virtual camera: rendering, OpenCV detection, projection, height of the ball
  metrics.py             Metrics: tracking error, return to the center, command jitter
  session.py             Sandbox: series of attempts, automatic end, CSV recording
  parallel.py            Independent simulations spread over several processes
tests/                   Automated tests, by theme (controller, mechanics, sensors, physics,
                         scenarios, metrics, session, regression check, campaign)
campaign.py              Full measurement campaign over N draws, in parallel -> results/report.md
sandbox.py               The robot in 3D and in real time, driven from a control panel
regression_check.py      Checks that a change leaves all the results bit-identical
```

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Usage

```powershell
python -m pytest              # all the tests
python campaign.py            # full measurement campaign, writes results/report.md (a few minutes)
python campaign.py --draws 1  # same, nominal configuration only (one simulation per case)
python sandbox.py             # 3D sandbox (see below)
python sandbox.py --camera                             # same, ball measured by the virtual camera
python -m simulation.mujoco_physics                    # exports the models to results/
python -m mujoco.viewer --mjcf=results/robot.xml       # opens the model alone in MuJoCo
python regression_check.py --record       # before a change: reference of all the results
python regression_check.py --compare      # after: checks that they are bit-identical
python regression_check.py --compare --by-position    # same after a renaming (changed names, same order)
```

Measurement campaign: every case is simulated over N draws (`--draws`, 10 by default), in parallel
(`--processes`, one per core by default). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise
seed k (`Config.seed`) and a start position shifted by up to ±0.5 mm on each axis, the only source of
variation for the camera, which draws no random number. The report gives mean ± standard deviation;
a difference between two cases, paired over the same draws, is significant by a paired t-test at the
5 % level (|mean| > t × standard error, t = 2.26 for 10 draws). Maximum speeds are found by bisection,
to 0.01 m/s. The report opens with the objectives of the brain (met or not); sections 0 and 10 validate
the simulator with a frozen reference controller (`simulation/config.py`: the hand-tuned PID without
latency compensation, error integrated everywhere, with the motors of the first robot).

Regression check: any refactoring must leave the results bit-identical. The reference
(`results/reference/`, ignored by git) covers the main cases of the campaign, the physics validation
and a sandbox sequence; every setting of these cases is written explicitly, so that changing a default
value does not change them. It is only valid on the machine and with the versions that produced
it (Windows and Linux diverge as soon as the ball bounces): the comparison warns if they have changed.

A simulation is described by a `Config`; everything that is not specified takes the reference value:

```python
from simulation.loop import simulate
from simulation.config import Config, center

r = simulate(Config(sensor="camera", latency=3))                  # the brain (default settings)
r = simulate(Config(sensor="camera", latency=3, latency_compensation=False))   # the brain without latency compensation
r = simulate(Config(physics="simple", mechanism=False))           # theoretical model
r = simulate(Config(trajectory=center, start_position=(-0.1, 0.0), start_velocity=(0.4, 0.0)))  # thrown ball
r = simulate(Config(start_height=0.305))                           # ball dropped from 30.5 cm (MuJoCo only)
```

## 3D sandbox

Two windows: the MuJoCo 3D view and a control panel with a top view and live curves.

| Control | Effect |
|---|---|
| Throw | Ball placed at the starting point (distance, angle), at the chosen height, thrown toward the center (optional deviation) |
| Drop | Same starting point, same height, without speed: the ball falls and bounces |
| Center | Ball placed motionless at the center |
| Free attempt | Records without touching the ball |
| Click in the top view | Chooses the starting point |
| Setpoint, latency, compensation | Change the behavior of the controller live |
| Space bar (3D view) or Pause | Pause / resume |
| Double-click on the ball then Ctrl + right drag (3D view) | Pushes the ball: a "disturbance" attempt starts by itself as soon as the ball moves more than 2 cm away |

An attempt ends when the ball falls, stays within 1 cm of the setpoint for 2 s (stabilized),
lasts longer than 30 s, or when another one starts. Each attempt is recorded in
`results/attempts/<date_time>/`: `attempt_001.csv`… (all the measurements at 50 Hz) and `summary.csv`
(one row per attempt: parameters, result, stabilization time, bounces, edge margin…).
International CSV format: `,` separator, decimal point, UTF-8 encoding (no BOM).

## Conventions

- SI units: meters, seconds, radians (degrees only for display).
- (x, y): position of the ball in the plate frame, origin at the center.
- θx > 0 lowers the +x side of the plate (the ball accelerates toward +x); same for θy.
- Plate rotation: R = Ry(θx)·Rx(-θy).

## Main assumptions (no real reference model)

| Element | Reference value |
|---|---|
| Plate | Disk of radius 15 cm, 0.25 kg; calibration error of the tilt 0.5° on θx, -0.3° on θy, unknown to the controller. At the start, the robot is already on: the plate holds the command 0 (hence the calibration error) and is at rest when the ball is placed |
| Ball | Ping-pong: radius 2 cm, 2.7 g, hollow (J = 2/3·m·r²) |
| Ball-plate contact | Target restitution 0.88 (ITTF rule: dropped from 30.5 cm, bounces back to 22-25 cm), stiffness 200 Hz, simulation step 0.5 ms |
| Mechanism | 3 arms of 5 cm, rods of 12 cm, ball joints 10 cm from the center. Without the arms (`mechanism=False`, angle-driven plate used to validate the simulator), the two angles of the plate follow the command with the profile of the motors, without steps |
| Motors | 10 rad/s, 1/16 microsteps, ramps at 50 rad/s² for the small corrections and 400 rad/s² for the large moves (gap above 0.1 rad, since B4; peak torque about 0.31 N·m, to check against a datasheet), the latter only when the brain allows it: for a ball faster than 0.3 m/s (a slow ball would be thrown off the plate), max torque 0.4 N·m; in MuJoCo, position servos (kp = 50 N·m/rad, kv = 0.5 N·m·s/rad) that lag their target by τ = kv/kp = 10 ms |
| Simple physics | Basic model ẍ = c·g·sin θ; optional additions (`Config`): rolling term r·θ̈, rotating-frame terms, exact gravity, servo lag. The full equation is derived in `simulation/simple_physics.py` and checked against MuJoCo; it explains the gap to MuJoCo on the angle-driven plate and on the robot (campaign, section 10) |
| Sensor | Camera 50 cm above the plate, 50 Hz, latency 20 ms (the first image arrives 20 ms after the start); height of the ball from its apparent size (± 3 mm), a ball partly out of the image is not measured |
| Controller (PID brain) | 50 Hz, gains by LQR (since B3: Kp = 1.89 rad/m, Ki = 1.75 rad/(m·s), Kd = 0.80 rad·s/m, from a position error of 20 cm, an integral over 0.5 s and a tilt of 10° of equal cost; before: Kp = 1, Ki = 1, ζ = 0.7), max tilt ±10°, latency compensation by prediction, error integrated only within 2 cm of the setpoint on each axis; ball in the air (measured above 1.5 cm): its parabola is fitted and the plate is tilted so that the bounce sends it halfway back to the setpoint (restitution assumed 0.87); at the first measurement, the plate is kept until the speed of the ball is known. All on by default |

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV, pytest.
