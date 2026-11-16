# PABLO

PABLO is a balancing robot ("ball and plate"), simulated (repository: ball-plate-sim): a plate tilted by 3 motors
keeps a ball balanced, catches it when it is thrown, and makes it follow a trajectory.
Goal: a piece of software for the general public, usable without any knowledge of code: a scientific sandbox
where one plays with the ball and compares two "brains" that drive the robot, a PID and a local AI. No game:
the statistics are only informative. The physics is computed by MuJoCo.
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
| C. Local AI brain | A policy (neural network) learned by reinforcement in MuJoCo, same rules as the PID (sensor, motors, ±10°). Done: interchangeable brains, training environment, training by reinforcement, imitation of the PID brain: the policy alone then meets the four objectives, dropped ball included (campaign section 13, which compares both brains: return to the center better than the PID, circle about 0.2 mm behind). The sandbox can be driven by either brain. Left for later: the gaps of more than 0.5 mm between the two brains explained (circle with a large calibration error of the plate, return to the center), then reinforcement from the imitating policy, to go beyond the PID | Done |
| D. Software for the general public | An application installed with a double click, for Windows and Mac: the robot in 3D, driven by either brain; throws, drops and pushes with the mouse, drawn trajectories, choice of the ball, latency and friction, both robots side by side; in English and French. Done: prototype (`python -m app`): 3D view in its own window, throw, drop, center, choice of the brain, pause, English and French; visual foundations: light and dark themes, panel of measurements; throws, drops and pushes with the mouse, physical limit of a throw, setpoints (center, circle, square); choice of the ball, friction and latency; both brains side by side, comparison of the brains; drawn trajectories; the robot shown as a lab prototype (motors, arms, rod ends, electronics, battery, camera under the plate, front panel with the LED of the brain), the ball printed according to its kind, the name PABLO | **In progress** |
| E. Construction (distant) | Design, real-world defects, complete error budget; first, simulate what the software shows and the campaign does not: the camera under the transparent plate (the campaign: 50 cm above it), the ball joints of the rods 1 cm under the plate surface (the physics: on it) | Later |

Quantified goals for both brains: thrown ball caught up to at least 90 % of the physical limit,
error on the circle < 2 mm with 60 ms of latency, return to the center < 2 s (from 10 cm away, the ball
stays within 5 mm of the center from 2 s on, whatever the calibration error of the plate from 0 to 1°),
no fall in the robustness variants.

## Structure

```text
app/                     The software for the general public (interface only, on top of simulation/session.py)
  window.py              Main window: control panel, 3D view (or two side by side); one loop in step with real time
  robot.py               A robot of the window: its session, its 3D view, its fallen ball followed by the camera
  comparison.py          Comparison of the brains on the attempts of the session
  view3d.py              3D view: rendered off screen by MuJoCo, drawn in the window with its texts; mouse: turn, zoom
  scene.py               The robot as shown: a copy of the physics model, dressed (materials, lights, floor, parts)
  parts.py               The parts of the robot shown: a lab prototype built around the simulated mechanism
  meshes.py              Small mesh library for the parts (lathe, prism, tube, text), MuJoCo binary format
  theme.py               Light and dark themes: colors, font, style sheet, colors of the 3D scene
  widgets.py             Building blocks: card, segmented control, readout, table, curves
  readouts.py            Measurements of the panel, read from the session
  gestures.py            Mouse gestures on the plate: throws, drops (fall along a parabola), pushes; physical limit of a throw
  benchmark.py           Cost of a frame on this computer: simulation, 3D rendering, drawing (python -m app.benchmark)
  i18n.py, locales/      Texts of the interface, one file per language (English, French)
  fonts/                 Inter, the font of the interface (SIL Open Font License, OFL.txt)
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
  metrics.py             Metrics: tracking error, return to the center, command jitter; physical limit of a throw
  session.py             Sandbox: series of attempts, automatic end, CSV recording; pace of real time
  parallel.py            Independent simulations spread over several processes
training/
  environment.py         Training environment of the AI brain: random episodes, reward, speed measurement
  imitate.py             Imitation: the policy learns to answer like the PID brain, its teacher
  train.py               Training of the policy by reinforcement (PPO), on several robots in parallel
  evaluation.py          Evaluation of a brain on the reference scenario, for both training scripts
tests/                   Automated tests, by theme (controller, mechanics, sensors, physics,
                         scenarios, metrics, session, regression check, campaign, training, display, software)
campaign.py              Full measurement campaign over N draws, in parallel -> results/report.md
sandbox.py               The robot in 3D and in real time, driven from a control panel (working tool)
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

To run the software for the general public (not needed for the rest: without it, its tests are skipped), add Qt
(PySide6, about 80 MB to download):

```powershell
pip install -r requirements-app.txt
```

## Usage

```powershell
python -m pytest              # all the tests
python campaign.py            # full measurement campaign, writes results/report.md (a few minutes)
python campaign.py --draws 1  # same, nominal configuration only (one simulation per case)
python -m app                 # the software for the general public (phase D, prototype)
python -m app.benchmark       # cost of a frame of the software on this computer (about a minute)
python sandbox.py             # 3D sandbox (see below)
python sandbox.py --camera                             # same, ball measured by the virtual camera
python sandbox.py --brain imitation                    # same, driven by the AI brain instead of the PID brain
python -m simulation.mujoco_physics                    # exports the models to results/
python -m mujoco.viewer --mjcf=results/robot.xml       # opens the model alone in MuJoCo
python regression_check.py --record       # before a change: reference of all the results
python regression_check.py --compare      # after: checks that they are bit-identical
python regression_check.py --compare --by-position    # same after a renaming (changed names, same order)
python -m training.environment            # speed of the training environment on this machine (1 minute)
python -m training.imitate --name pupil --rounds 8 --episodes 1500    # the policy of the AI brain learns to
                                                       # imitate the PID brain (less than an hour)
python -m training.train --name first --periods 2e5    # training by reinforcement: a trial of a few minutes
python -m training.train --name first --periods 5e7    # same training, continued up to 50 million periods
```

Imitation: rounds of episodes of 10 s (`--rounds`, `--episodes`: 6 and 1000 by default), on half of the
logical cores (`--processes`), by a network of 2 layers of 128 neurons (`--neurons`). After every round, the
pupil alone is compared with the PID brain alone: on the reference scenario, on 600 episodes that are never
learned (falls counted apart for the balls dropped from above), and on the reference scenario seen by the
camera, which no lesson uses. The network is written in `results/training/<name>/policy.npz` (ignored by git);
copied to `brains/imitation.npz`, it becomes the AI brain of the simulations.

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
128 neurons (`controller/network.py`), learned in simulation (phase C). The observation holds what
the PID brain also knows (`controller/policy.py`): the last 8 measurements and the last 12 commands, raw (the
network filters their noise and predicts the ball despite the latency by itself; the measured heights show a
ball in the air falling, and count as 0 below 15 mm, where they are only the error of the sensor), the setpoint
with its speed
and its acceleration, the latency, and two long memories kept like the PID keeps its own: the integral of the
error to the setpoint (a calibration error of the plate shows in nothing else) and an estimate of the response
of the ball, c·g (`controller/adaptation.py`: 8 measurements are too short to tell a solid ball from a hollow
one, and the plate must not be tilted the same for both). The action is the command (θx, θy) in units of the maximum tilt. Once
learned, the network drives alone: the PID brain is no longer there.

Learned in two ways. By imitation first (`training/imitate.py`): the PID brain is the teacher. It drives
episodes while the observations of the pupil and its own commands are recorded, and the network learns to give
these commands; then the pupil drives, the teacher only telling what it would have done, so that the pupil also
learns to recover from its own errors. The episodes are those of the reinforcement (below), one out of 4 with
a ball dropped from above: the pupil has no estimate of the flight of its own, it learns from the PID how to
tilt the plate under a ball in the air. Measured after 8 rounds of 1500 episodes (5.6 million periods), the
pupil alone next to the PID brain alone:

| | PID brain | Imitating policy |
|---|---|---|
| Return to the center, 20 ms / 60 ms of latency | 1.21 / 1.15 mm | 0.66 / 0.64 mm |
| Circle error, 20 ms / 60 ms | 0.79 / 1.00 mm | 0.89 / 0.98 mm |
| Tremor of the command on the circle, 20 ms / 60 ms | 0.11° / 0.19° | 0.11° / 0.19° |
| Falls over 600 episodes of the training, never learned | 6.7 % | 6.5 % |
| of which the 149 balls dropped from above | 0 | 0 |

(one draw of the sensor noise; section 13 of the campaign compares them over all its draws, with the camera:
the four objectives are met by both.) This policy is saved in `brains/imitation.npz`. Without the estimate of
the ball, the policy tilted the plate for an average ball: on the circle, 0.7 mm too far out with the hollow
ball, 1.1 mm too far in with the solid one (1.10 and 1.49 mm of circle error in the campaign, 0.97 and 0.87 mm
with the estimate; PID: 0.78 and 0.68 mm). A network of 2 layers of 256 neurons did not imitate better. Then by reinforcement, where the
policy improves by itself, by trial. From scratch, 10 million periods of reinforcement had left it far from the
PID (return 3 to 8 mm, circle 3 to 10 mm, a command trembling 3 to 5 times more): it starts again from the imitating
policy. The training environment of the reinforcement (`training/environment.py`)
plays episodes of 10 s on the MuJoCo robot, each with another robot, ball, sensor and scenario drawn at random
(inertia, mass and friction of the ball, calibration error, latency, noise; ball placed, thrown, pushed or
dropped from above;
setpoint fixed, jumping or circling), and rewards every period, from the distance d to the setpoint: half for
staying near it, 1 - (d / 10 cm)², half for the last millimeters, 1 / (1 + d / 5 mm), minus the jerk of the
command, and a penalty if the ball falls. A policy sees exactly the same robot in the environment and in a
simulation (checked bit for bit in the tests).

The reinforcement (`training/train.py`) uses PPO (proximal policy optimization, from Stable-Baselines3):
during the training, a random deviation is added to the action of the policy, to try. The robots play 20 s each, then the actions that brought more reward than expected are made more likely,
by small steps, millions of times.

## Software for the general public (phase D, in progress)

`python -m app`: one window, the control panel on the left, the robot in 3D on the right; or, side by side, two
robots, one per brain (PID on the left, AI on the right), each with its own simulation: every action (a gesture on
either view, the setpoint, a setting, Center) is applied to both, the views turn and zoom together, and the two
simulations advance step by step together.

| Panel | Content |
|---|---|
| Brain | PID or AI; changing it ends the current attempt (an attempt has a single brain). Side by side: both brains, a robot each; when it ends, the first robot stays |
| Ball | Tools Throw, Drop and Push (active until clicked again), Center, Pause; the height of a drop (0 to 40 cm; the AI brain saw 5 to 35 cm in training, said outside them) |
| Setpoint | Center, circle (6 cm in 4 s), square (10 cm sides at 5 cm/s) or Drawing: where the brain must hold the ball. A new one starts an attempt "new setpoint". Drawing: the tool Draw; the path drawn on the plate closes by itself in a straight line and is traveled in a loop at the speed chosen (2 to 10 cm/s). It is resampled every mm and smoothed over 1.5 cm (the hand trembles), its acceleration over 4 cm (measured on a 6 cm circle drawn with 0.5 mm of tremble: within 2 mm of the circle setpoint, its acceleration within 6 %). A path farther than 10 cm from the center is refused (the ball keeps 3 cm from the edge). The AI brain never saw a drawn path in training (points, steps, circles) |
| Settings of the robot | Ball: ping-pong (hollow, 2.7 g), solid (J = 2/5·m·r²) or heavy (hollow, 27 g); friction ball-plate μ = 1, 0.2 or 0.02; latency of the camera, 20 to 100 ms. As in the robustness variants of the campaign, and all within what the AI brain saw in training (the comparison stays fair; 100 ms is also the most it can take). Changing one ends the current attempt. The ball and the friction are changed in the MuJoCo model in place, exactly as if it had been built with them (a test checks it, to the last bit) |
| Measurements | Distance to the setpoint, speed and height of the ball, tilt of the plate (x, y), largest torque of the 3 motors; the distance over the last 10 s. True values of the simulation, not those measured by the sensor of the robot. Side by side: a column per brain, a curve per brain, in its color |
| Last attempt | Its action, its brain and its result (stabilized in, fell after...); its ball, friction and latency. Side by side: that of each robot |
| Comparison | By brain, over the finished attempts of the session made in the current conditions (setpoint, the same drawn path at the same speed, ball, friction, latency), side by side or not: number of attempts, share of the balls caught, median time to stabilize, median room left to the edge of the plate. Counted: throws, drops, pushes, disturbances and new setpoints that came to a result (not interrupted). The room to the edge rather than the largest distance to the setpoint: for a throw, that one mostly tells where the ball was thrown from |
| Bottom | Light or dark theme (that of the system at startup), English or French (that of the system). The panel is at least 300 pixels wide, wider if a row needs it (texts and fonts change with the language and the system): nothing of it is cut or under its scroll bar |

Over the 3D view: what the robot is doing, the position of the ball, the simulated time with its speed compared
with real time and the frames per second, the x and y axes of the plate (they turn with the view), the setpoint
and its path. The attempts follow the rules of the sandbox below, without being recorded. A view narrower than it
is tall (side by side) moves its camera back, to keep the width of the scene in sight. Each view has its own MuJoCo
renderer; MuJoCo frees a renderer in the OpenGL context that is current, so each is freed in its own (otherwise,
measured, the other view drew garbage after a resize).

Mouse: the wheel zooms; a drag turns the view, unless a tool is active and the drag starts on the plate:

| Tool | Gesture | Effect |
|---|---|---|
| Throw | Drag from where the ball starts | The ball starts there, rolling along the arrow. Speed = length of the arrow / 0.2 s (10 cm: 0.5 m/s), up to 1.5 m/s. While dragging: the speed, and the physical limit of this throw √(2·c·g·sin θmax·d), d the distance to the edge along the arrow (nominal ball); green within it, red beyond (allowed, to see the ball fall) |
| Drop | Click where the ball falls, or drag from there | Click: the ball is dropped from the height chosen, without speed. Drag: it is released above the point pressed with the horizontal speed of the arrow (same scale), and falls along a parabola; while dragging, its fall, where it lands and its angle of arrival from the vertical (green if it lands on the plate, red if not; no limit shown: not computed, bounce then rolling, so on the plate does not mean caught) |
| Draw | Drag a path on the plate (setpoint Drawing) | It becomes the setpoint; drawn in green within 10 cm of the center, in red beyond (refused) |
| Push | Drag from the ball | Its velocity changes by the arrow (same scale); the brain keeps its memory |

When the ball falls, the camera glides to look between the robot and the ball and moves back to keep both in view
(up to 2.5 m away); after 2 s of simulated time, the ball comes back to the center of the plate, and the camera to
where it was. The pause stops all of it.

The fall drawn for a drop is computed for the plate as it is when the arrow is drawn (no air in the model: a
parabola, `app/gestures.drop_flight`). With the plate held still, MuJoCo lands the ball within 0.2 mm of it (one
step of the physics); the brain moves the plate during the fall, so the ball actually lands a few mm away (measured:
0.1 to 3 mm on the plate). Neither brain has learned drops with a speed (the training drops the ball straight down).

The point under the mouse comes from the camera (`app/view3d.Camera`): the ray of the pixel, computed as MuJoCo
computes its free camera, meets the plane of the plate surface. A test checks that the pixel computed for the
center of the ball is where MuJoCo draws it (within 2 pixels).

The robot shown is a copy of the MuJoCo model of the physics, dressed for display (`app/scene.py`): materials,
lights, a floor that continues the background of the window, and the parts of a lab prototype (`app/parts.py`): an
aluminium base on rubber pads; per leg, a NEMA 17 stepper motor on its bracket, a machined arm, a threaded rod
between two rod ends, a bracket under the plate; the camera under the transparent plate, looking up, with its ring of
LEDs; a small computer, a board of three stepper drivers, a LiPo battery (no cable leaves the robot); a front panel
with the name, PABLO (set in one place: `app/__init__.py`), and two LEDs, PID and AI (in the language of the
interface), the one of the brain lit in green; on the ball, its seam and a printing according to its kind (40+ ★★★
for the ping-pong ball, 2/5 mr² for the solid one, 27 g for the heavy one), which shows it turning, rolling or
sliding. At every frame, the positions of the physics are copied into it and its geometry is computed, without
dynamics. The moving parts are carried by the bodies of the physics, on its joints (the arm turns about the motor
axis and ends at the elbow, the rod is as long between its rod ends as the simulated one), so the picture shows the
robot that is simulated; the simple shapes of the physics stay in the model, hidden. One liberty: the upper rod ends
sit 10.5 mm under the plate surface, where the physics puts the ball joints (no real part can). When a rod leans on
the plate by an angle a, the rod end and its bolt are 2 × 10.5 mm × sin(a/2) apart: 1.8 mm at 10° (measured over four
hard catches: a up to 15.6°, 2.9 mm). The picture also shows the give of the simulated ball joints: 0.1 mm at rest,
up to 1.9 mm in a hard catch (measured). In a hard catch an arm goes down by 28° (measured): the base is low enough
for 30°, its lowest part stays 1.7 mm above it. The camera under the plate is only shown: the sandbox measures the
ball with the synthetic sensor, and the campaign validated a camera 50 cm above the plate. The parts are meshes built
once with NumPy (0.35 s on the development computer), then shared by every scene (themes, side by side): 45 000
triangles. The physics, and the virtual camera that renders it, see nothing of this: the parts touch nothing (checked
by the tests and the bit-exact regression check). Quality of the rendering, from the measurements of `python -m
app.benchmark` on an integrated graphics card (Intel Arc), with the parts of the prototype: a frame costs 15.6, 25.0
and 39.0 ms at 1280×720, 1920×1080 and 2560×1440 (58, 37 and 24 frames per second), as much as the bare physics model
of phase D1 (16.3, 23.4 and 41.8 ms): the cost is in the number of pixels, not in the parts. Antialiasing with 4
samples (8 cost 35 to 67 % more), shadow map of 4096 pixels and fog (without shadows, nothing measurable is saved;
without fog or with a map of 2048 pixels, the frame came out slower, 23 to 52 ms: the measurements vary from one run
to the next). The shadows and the fog follow the view (`DisplayScene.view`): in the default view, the shadow map
covers 0.72 m around the robot (0.35 mm per pixel of it: sharp edges) and the fog fades the floor into the sky from
1.2 to 3.6 m from the eye; zoomed out, or following a fallen ball, these lengths grow with the distance of the camera
and the map is centered on the point looked at, so the floor keeps its look. Measured: image unchanged in the default
view; dots on the floor beyond the map 2.3 % → 0.2 % zoomed out, 3.6 % → 0.2 % following a fallen ball (one map 6 m
wide for every view gave steps on the edges of the shadows, a far fog a sharp horizon). The floor casts no shadow (it
receives those of the robot, `DisplayScene.prepare`): drawn into the shadow map, it shadowed itself in dots wherever
the camera saw it beyond the map, in a region bounded by a corner of the map (measured at azimuth 0°: 0.43 % of the
floor seen; none now). MuJoCo counts these lengths in sizes of the scene (1.2 m); they are changed in its rendering
context, an internal attribute of `mujoco.Renderer` (a test checks it). In the light theme the plate is smoked
acrylic: a clear one hardly stood out from the white background (measured: the plate changes the lightness of what is
seen through it by 23 units of L* against 13 for the clear one; 29 in the dark theme). The orange of the ball marks
what is selected (brain, theme, language, pause); the buttons that act are neutral.

`python -m app.benchmark` measures what a frame costs on the computer where it runs: the simulation, the 3D
rendering (with the image read back) for three sizes of the view and several qualities, the drawing of the image
by Qt; then the frames per second they allow together. It serves to choose the quality of the rendering, and to
check the software on other computers.

A single loop, in the thread of the interface: at every frame (60 per second), the simulation catches up with the
real time elapsed (at most 100 ms: beyond, it slows down), then MuJoCo renders the 3D view off screen, and the
image is drawn in the window. The virtual camera, which also renders with MuJoCo, will need it: its OpenGL context
can only be used in the thread that created it.

## 3D sandbox

The working tool of the development, kept until the software does all it does. Two windows: the MuJoCo 3D view
and a control panel with a top view and live curves. `--brain imitation`: the robot is driven by the AI brain (it
has no latency compensation to switch off).

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

An attempt ends when the ball falls, stays within 1 cm of the setpoint for 2 s (stabilized), lasts longer than 30 s,
when another one starts (a new setpoint starts one too), or when a setting changes (brain, latency, compensation; in
the software, also the ball and the friction): an attempt has a single setting of each. A fallen ball lands on the
ground, which touches nothing else. Each attempt is recorded in `results/attempts/<date_time>/`: `attempt_001.csv`…
(all the measurements at 50 Hz) and `summary.csv` (one row per attempt: parameters, setpoint and drawn path, brain,
ball, friction, latency, result, stabilization time, bounces, edge margin…).
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

Python, NumPy, Matplotlib, MuJoCo, OpenCV, pytest. For the software: Qt (PySide6), the Inter font. For the training
of the AI brain only: PyTorch, Stable-Baselines3.
