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
| B. PID brain | Objectives in the report, integral limited to the neighborhood of the setpoint, aimed bounce of a dropped ball, thrown ball (start with the speed, fast motor ramps for the large moves), gains tuned by LQR, adaptive control (model gain c·g estimated online), every gap of more than 0.5 mm explained (report, section 12) | Done |
| C. Local AI brain | A policy (neural network) learned by reinforcement in MuJoCo, same rules as the PID (sensor, motors, ±10°). Done: interchangeable brains, training environment, training by reinforcement, imitation of the PID brain (the policy alone then does as well as the PID on the reference scenario). The campaign compares both brains (section 13) and the sandbox can be driven by either. To come: reinforcement from the imitating policy, to go beyond the PID | **In progress** |
| D. Software for the general public | MuJoCo physics in our interface, choice of brain, comparison | To do |
| E. Construction (distant) | Design, real-world defects, complete error budget | Later |

Quantified goals for both brains: thrown ball caught up to at least 90 % of the physical limit,
error on the circle < 2 mm with 60 ms of latency, return to the center < 2 s (from 10 cm away, the ball
stays within 5 mm of the center from 2 s on, whatever the calibration error of the plate from 0 to 1°),
no fall in the robustness variants.

## Structure

```text
brains/                  Learned policies of the AI brain (weights of the network): imitation.npz
controller/              What would run on the real robot (independent of the simulator)
  pid.py                 PID: derivative filter, saturation, anti-windup, integration zone, feedforward
  plate.py               PID brain: 2 PIDs (x, y axes) + feedforward + latency predictor + aimed bounce
  adaptation.py          Adaptive control: online estimation of the model gain c·g (inertia of the ball)
  policy.py              AI brain: memory and observation of a brain driven by a policy (observation -> action)
  network.py             Neural network of the AI brain, computed with NumPy alone (no PyTorch needed to use it)
  ramps.py               Rule of the fast ramps of the motors, the same for every brain
  flight.py              Ball in the air: ballistic tracking, prediction of the impact
  kinematics.py          Geometry, inverse (exact) and forward (Newton) kinematics
simulation/
  config.py              Single configuration, physical parameters, setpoint trajectories
  loop.py                Single closed loop, whatever the brain: sensor -> brain -> motors -> physics
  simple_physics.py      Simple physics (theoretical reference)
  mujoco_physics.py      MuJoCo physics: generation of the 3D models, contact, interface, validation
  motor.py               Stepper motor: max speed, microsteps, limits, ramps
  sensor.py              Synthetic sensor: noise + latency
  camera.py              Virtual camera: rendering, OpenCV detection, projection, height of the ball
  metrics.py             Metrics: tracking error, return to the center, command jitter
  session.py             Sandbox: series of attempts, automatic end, CSV recording
  parallel.py            Independent simulations spread over several processes
training/
  environment.py         Training environment of the AI brain: random episodes, reward, speed measurement
  imitate.py             Imitation: the policy learns to answer like the PID brain, its teacher
  train.py               Training of the policy by reinforcement (PPO), on several robots in parallel
  evaluation.py          Evaluation of a brain on the reference scenario, for both training scripts
tests/                   Automated tests, by theme (controller, mechanics, sensors, physics,
                         scenarios, metrics, session, regression check, campaign, training)
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

To train the AI brain (not needed to simulate, nor to run the tests: those of the training script are then
skipped), add the training libraries, PyTorch (about 130 MB to download) and Stable-Baselines3:

```powershell
pip install -r requirements-train.txt
```

## Usage

```powershell
python -m pytest              # all the tests
python campaign.py            # full measurement campaign, writes results/report.md (a few minutes)
python campaign.py --draws 1  # same, nominal configuration only (one simulation per case)
python sandbox.py             # 3D sandbox (see below)
python sandbox.py --camera                             # same, ball measured by the virtual camera
python sandbox.py --brain imitation                    # same, driven by the AI brain instead of the PID brain
python -m simulation.mujoco_physics                    # exports the models to results/
python -m mujoco.viewer --mjcf=results/robot.xml       # opens the model alone in MuJoCo
python regression_check.py --record       # before a change: reference of all the results
python regression_check.py --compare      # after: checks that they are bit-identical
python regression_check.py --compare --by-position    # same after a renaming (changed names, same order)
python -m training.environment            # speed of the training environment on this machine (1 minute)
python -m training.imitate --name pupil   # the policy of the AI brain learns to imitate the PID brain (20 minutes)
python -m training.train --name first --periods 2e5    # training by reinforcement: a trial of a few minutes
python -m training.train --name first --periods 5e7    # same training, continued up to 50 million periods
```

Imitation: 6 rounds of 1000 episodes of 10 s (`--rounds`, `--episodes`), on half of the logical cores
(`--processes`). After every round, the pupil alone is compared with the PID brain alone, on the reference
scenario and on 300 episodes that are never learned. The network is written in
`results/training/<name>/policy.npz` (ignored by git).

Reinforcement: `--periods` is the total of control periods (20 ms of the robot each), over all the robots simulated
in parallel: `--processes` processes (half of the logical cores by default) of `--per-process` robots each (4 by
default). Every million periods (`--every`), the training is saved in `results/training/<name>/` (ignored by
git) and the policy is evaluated on the reference scenario: one line printed, one row in `progress.csv`, and a
second line telling where the time went (simulation, exchanges with the robots, policy, learning, evaluation).
Ctrl+C saves and stops; the same command resumes, on as many robots.

Measurement campaign: every case is simulated over N draws (`--draws`, 10 by default), in parallel
(`--processes`, one per core by default). Draw 0 is the nominal configuration; draw k ≥ 1 uses noise
seed k (`Config.seed`) and a start position shifted by up to ±0.5 mm on each axis, the only source of
variation for the camera, which draws no random number. The report gives mean ± standard deviation;
a difference between two cases, paired over the same draws, is significant by a paired t-test at the
5 % level (|mean| > t × standard error, t = 2.26 for 10 draws). Maximum speeds are found by bisection,
to 0.01 m/s. The report opens with the objectives of the brain (met or not); sections 0 and 10 validate
the simulator with a frozen reference controller (`simulation/config.py`: the hand-tuned PID without
latency compensation, error integrated everywhere, with the motors of the first robot), over 4 times more
draws (the noise of the synthetic sensor makes their return vary by more than 1 mm); section 12 explains
every other gap of more than 0.5 mm; section 13 puts the AI brain next to the PID brain, on the same objectives,
cases and draws.

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

## The two brains

A brain receives, every 20 ms, the measurement of the ball (late by the latency), the setpoint and its
acceleration, and answers with the two tilts of the plate. The loop (`simulation/loop.py`) accepts any brain:
`Config(brain="pid")`, the default, chooses the PID brain; `Config(brain="imitation")` the AI brain, driven by
the learned policy `brains/imitation.npz`; `simulate(cfg, brain=policy_brain(policy))` any other policy.
The motors, their ramps and the rule of the fast ramps are the same for both.

The policy of the AI brain is a function from an observation to an action: a neural network of 2 layers of
128 neurons (`controller/network.py`), learned in simulation (phase C, in progress). The observation holds what
the PID brain also knows (`controller/policy.py`): the last 8 measurements and the last 12 commands, raw (the
network filters their noise and predicts the ball despite the latency by itself), the setpoint with its speed
and its acceleration, the latency, the error to the setpoint and its integral (the only long memory of the
brain, kept like the PID keeps its own). The action is the command (θx, θy) in units of the maximum tilt. Once
learned, the network drives alone: the PID brain is no longer there.

Learned in two ways. By imitation first (`training/imitate.py`): the PID brain is the teacher. It drives
episodes while the observations of the pupil and its own commands are recorded, and the network learns to give
these commands; then the pupil drives, the teacher only telling what it would have done, so that the pupil also
learns to recover from its own errors. Measured after 6 rounds (2.8 million periods, 19 minutes on 11 processes),
the pupil alone next to the PID brain alone:

| | PID brain | Imitating policy |
|---|---|---|
| Return to the center, 20 ms / 60 ms of latency | 1.21 / 1.15 mm | 0.97 / 0.66 mm |
| Circle error, 20 ms / 60 ms | 0.79 / 1.00 mm | 1.11 / 1.07 mm |
| Tremor of the command on the circle, 20 ms / 60 ms | 0.11° / 0.19° | 0.11° / 0.19° |
| Falls over 300 episodes of the training, never learned | 8.7 % | 8.3 % |

(one draw of the sensor noise; section 13 of the campaign compares them over all its draws, with the camera.)
This policy is saved in `brains/imitation.npz`. Then by reinforcement, where the
policy improves by itself, by trial. From scratch, 10 million periods of reinforcement had left it far from the
PID (return 3 to 8 mm, circle 3 to 10 mm, a command trembling 3 to 5 times more): it starts again from the imitating
policy. The training environment of the reinforcement (`training/environment.py`)
plays episodes of 10 s on the MuJoCo robot, each with another robot, ball, sensor and scenario drawn at random
(inertia, mass and friction of the ball, calibration error, latency, noise; ball placed, thrown or pushed;
setpoint fixed, jumping or circling), and rewards every period, from the distance d to the setpoint: half for
staying near it, 1 - (d / 10 cm)², half for the last millimeters, 1 / (1 + d / 5 mm), minus the jerk of the
command, and a penalty if the ball falls. A policy sees exactly the same robot in the environment and in a
simulation (checked bit for bit in the tests).

The reinforcement (`training/train.py`) uses PPO (proximal policy optimization, from Stable-Baselines3):
during the training, a random deviation is added to the action of the policy, to try. The robots play 20 s each, then the actions that brought more reward than expected are made more likely,
by small steps, millions of times.

## 3D sandbox

Two windows: the MuJoCo 3D view and a control panel with a top view and live curves. `--brain imitation`: the
robot is driven by the AI brain (it was not taught the ball dropped from above, and has no latency compensation
to switch off).

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
| Ball-plate contact | Target restitution 0.88 (ITTF rule: dropped from 30.5 cm, bounces back to 22-25 cm), stiffness 200 Hz, simulation step 0.5 ms; friction μ = 1, Coulomb's round friction cone (`cone="elliptic"`, since the end of phase B; MuJoCo's own default, a pyramid, allows up to 29 % less friction between its axes when the ball slides) |
| Mechanism | 3 arms of 5 cm, rods of 12 cm, ball joints 10 cm from the center. Without the arms (`mechanism=False`, angle-driven plate used to validate the simulator), the two angles of the plate follow the command with the profile of the motors, without steps |
| Motors | 10 rad/s, 1/16 microsteps, ramps at 50 rad/s² for the small corrections and 400 rad/s² for the large moves (gap above 0.1 rad, since B4; peak torque about 0.31 N·m, to check on the torque curve of the chosen motor before construction: a 17HS4401S holds 0.40 N·m at rest, less at 10 rad/s), the latter only for a ball measured faster than 0.3 m/s, whatever the brain (a slow ball would be thrown off the plate), max torque 0.4 N·m; in MuJoCo, position servos (kp = 50 N·m/rad, kv = 0.5 N·m·s/rad) that lag their target by τ = kv/kp = 10 ms |
| Simple physics | Basic model ẍ = c·g·sin θ; optional additions (`Config`): rolling term r·θ̈, rotating-frame terms, exact gravity, servo lag, sliding (Coulomb friction). The full equation is derived in `simulation/simple_physics.py` and checked against MuJoCo; it explains the gap to MuJoCo on the angle-driven plate and on the robot (campaign, section 10), and with sliding, the remaining gaps (section 12) |
| Sensor | Camera 50 cm above the plate, 50 Hz, latency 20 ms (the first image arrives 20 ms after the start); height of the ball from its apparent size (± 3 mm), a ball partly out of the image is not measured |
| Controller (PID brain) | 50 Hz, gains by LQR (since B3: Kp = 1.89 rad/m, Ki = 1.75 rad/(m·s), Kd = 0.80 rad·s/m, from a position error of 20 cm, an integral over 0.5 s and a tilt of 10° of equal cost; before: Kp = 1, Ki = 1, ζ = 0.7), max tilt ±10°, latency compensation by prediction, error integrated only within 2 cm of the setpoint on each axis; ball in the air (measured above 1.5 cm): its parabola is fitted and the plate is tilted so that the bounce sends it halfway back to the setpoint (restitution assumed 0.87); at the first measurement, the plate is kept until the speed of the ball is known; model gain c·g of the feedforward and of the latency predictor estimated online (since B6: recursive least squares on the measured accelerations and the commands, only while the ball rolls within 1 cm of its setpoint; from the hollow ball at every attempt). All on by default |

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV, pytest. For the training of the AI brain only: PyTorch, Stable-Baselines3.
