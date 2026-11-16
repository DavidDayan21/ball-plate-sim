"""Single configuration of all simulations, and setpoint trajectories.

All physical parameters are initial assumptions (no real reference model):
they can be changed to explore other designs.
"""
from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

from controller.kinematics import Geometry
from controller.lqr import pid_gains

G = 9.81  # m/s²


# --- Physical parameters -------------------------------------------------------------------
@dataclass(frozen=True)
class PlateParams:
    plate_radius: float = 0.15    # circular plate (m)
    ball_radius: float = 0.02     # (m)
    k_inertia: float = 2 / 3      # J = k·m·r²: 2/3 hollow ball (ping-pong), 2/5 solid ball
    bias_x: float = 0.0           # calibration error on θx (rad), unknown to the controller
    bias_y: float = 0.0           # calibration error on θy (rad), unknown to the controller

    @property
    def coefficient(self) -> float:
        """c = 1/(1+k), in ẍ = c·g·sin θ."""
        return 1.0 / (1.0 + self.k_inertia)


@dataclass(frozen=True)
class MujocoParams:
    """Parameters added by the MuJoCo physics."""
    ball_mass: float = 0.0027          # ping-pong ball (kg)
    plate_mass: float = 0.25           # acrylic disk Ø30 cm, 3 mm (kg, estimate)
    plate_thickness: float = 0.003     # (m)
    arm_mass: float = 0.02             # (kg)
    rod_mass: float = 0.01             # (kg)
    kp_servo: float = 50.0             # stiffness of the position servo (N·m/rad)
    kv_servo: float = 0.5              # damping of the servo (N·m·s/rad)
    max_torque: float = 0.4            # max torque of a motor (N·m, order of magnitude of a NEMA 17)
    armature: float = 5e-6             # rotor inertia (kg·m²)
    restitution: float = 0.88          # target ball-plate coefficient of restitution. ITTF rule:
                                       # dropped from 30.5 cm, the ball bounces back to 22-25 cm, i.e.
                                       # e = √(h2/h1) between 0.85 and 0.91 (air losses included)
    contact_frequency: float = 200.0   # contact stiffness, expressed as a natural frequency (Hz).
                                       # Requires dt_physics ≤ 0.5 ms (otherwise spurious bounces)
    friction: float = 1.0              # ball-plate; MuJoCo keeps the larger of the two geoms,
                                       # so it is set on both
    k_actual: float | None = None      # actual inertia of the ball; None: the one the controller assumes.
                                       # Also used by the simple physics
    cone: str = "elliptic"             # friction cone of the contacts: "elliptic" (Coulomb's law, since the end of
                                       # phase B) or "pyramidal" (MuJoCo's own default: 4 facets, up to 29 % less
                                       # friction between its axes when the ball slides)


@dataclass(frozen=True)
class CameraParams:
    height: float = 0.50               # above the plate center (m)
    fovy: float = 40.0                 # vertical field of view (°)
    width: int = 320                   # resolution (pixels)
    image_height: int = 240
    hue: tuple = (5, 35)               # OpenCV hue range (0-179) of the orange ball
    saturation_min: int = 100
    value_min: int = 80
    pixels_min: int = 20               # below this: ball not detected


# --- Setpoint trajectories: t -> (x_ref, y_ref, ax_ref, ay_ref) ----------------------------------
@dataclass(frozen=True)
class Point:
    """Ball held at (x, y)."""
    x: float = 0.0
    y: float = 0.0

    def __call__(self, t: float):
        return self.x, self.y, 0.0, 0.0


@dataclass(frozen=True)
class Step:
    """Ball held at `start` (x, y), then at `end` from the time `time` on."""
    start: tuple
    end: tuple
    time: float

    def __call__(self, t: float):
        x, y = self.start if t < self.time else self.end
        return x, y, 0.0, 0.0


@dataclass(frozen=True)
class Circle:
    """Circle of radius `radius` around (cx, cy), traveled in `period` seconds (negative: clockwise), starting at
    the angle `phase`."""
    radius: float
    period: float
    phase: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def __call__(self, t: float):
        w = 2 * math.pi / self.period
        phi = w * t + self.phase
        x, y = self.radius * math.cos(phi), self.radius * math.sin(phi)
        return self.cx + x, self.cy + y, -w ** 2 * x, -w ** 2 * y


center = Point()                                                  # ball held at the center
circle = Circle(radius=0.06, period=4.0, phase=math.pi / 4)       # 6 cm in 4 s (9.4 cm/s), starting at 45°


def reference(t: float):
    """Reference scenario. 0-4 s: center. 4-8 s: point (5 cm, 5 cm).
    Then: circle of radius 6 cm traveled in 4 s, starting at 45°."""
    if t < 4.0:
        return 0.0, 0.0, 0.0, 0.0
    if t < 8.0:
        return 0.05, 0.05, 0.0, 0.0
    return circle(t - 8.0)


def square(t: float):
    """Square with 10 cm sides at 5 cm/s (2 s per side), starting at the corner (5 cm, 5 cm)."""
    corners = [(0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05)]
    k, s = int(t // 2.0) % 4, (t % 2.0) / 2.0
    (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 0.0, 0.0


@dataclass(frozen=True, eq=False)
class DrawnPath:
    """Closed path through `points` ((x, y), as drawn by hand), traveled at `speed` (m/s) from the first point, back
    to it in a straight line, and again. The points are resampled every PATH_STEP of length, smoothed over about
    PATH_SMOOTH (the hand trembles), and resampled again: the path is then traveled at constant speed. Its
    acceleration is that of the smoothed path, speed² × its second derivative along it (taken over PATH_SPAN, then
    smoothed over PATH_SMOOTH_CURVATURE: a tremble of the hand weighs much more on it): on a circle, v²/R toward the
    center, as Circle gives it."""
    points: tuple
    speed: float

    def __post_init__(self):
        drawn = np.asarray(self.points, dtype=float)
        loop = _resample(drawn, PATH_STEP)
        loop = _resample(_smooth(loop, PATH_SMOOTH), PATH_STEP)
        h = max(1, round(PATH_SPAN / PATH_STEP))
        curvature = (np.roll(loop, -h, axis=0) - 2 * loop + np.roll(loop, h, axis=0)) / (h * PATH_STEP) ** 2
        object.__setattr__(self, "samples", loop)
        object.__setattr__(self, "accelerations", self.speed ** 2 * _smooth(curvature, PATH_SMOOTH_CURVATURE))
        object.__setattr__(self, "length", len(loop) * PATH_STEP)

    def __call__(self, t: float):
        u = (self.speed * t % self.length) / PATH_STEP
        i = int(u) % len(self.samples)
        j, f = (i + 1) % len(self.samples), u - int(u)
        x, y = (1 - f) * self.samples[i] + f * self.samples[j]
        ax, ay = (1 - f) * self.accelerations[i] + f * self.accelerations[j]
        return float(x), float(y), float(ax), float(ay)

    @property
    def key(self) -> str:
        """Short name of this path and speed, to tell the attempts made on it from others."""
        rounded = repr((np.round(np.asarray(self.points), 4).tolist(), round(self.speed, 4)))
        return hashlib.sha1(rounded.encode()).hexdigest()[:8]


PATH_STEP = 0.001       # (m) between the samples of a drawn path
PATH_SMOOTH = 0.015     # (m) length of the smoothing window
PATH_SPAN = 0.010       # (m) on each side, for the second derivative
PATH_SMOOTH_CURVATURE = 0.040   # (m) length of the smoothing window of the second derivative


def _smooth(values: np.ndarray, length: float) -> np.ndarray:
    """Moving average of the samples of a closed path (every PATH_STEP) over about `length`, applied twice
    (triangular weights)."""
    n = max(1, round(length / PATH_STEP)) | 1                 # odd window
    for _ in range(2):
        values = sum(np.roll(values, k, axis=0) for k in range(-(n // 2), n // 2 + 1)) / n
    return values


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` of length along the polyline `points`, closed (back to the first point); the last
    sample is short of the first one."""
    closed = np.vstack([points, points[:1]])
    lengths = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(lengths)])
    keep = np.concatenate([[True], lengths > 1e-12])          # repeated points add nothing
    s, closed = s[keep], closed[keep]
    at = np.arange(0.0, s[-1] - step / 2, step)
    return np.column_stack([np.interp(at, s, closed[:, 0]), np.interp(at, s, closed[:, 1])])


# --- Configuration of a simulation --------------------------------------------------------------
@dataclass
class Config:
    # Physics
    physics: str = "mujoco"            # "simple" (our own equations) or "mujoco"
    mechanism: bool = True             # True: 3 motors + kinematics; False: angles applied directly
    # Simple physics: additions to the basic model (simulation/simple_physics.py), all off by default
    rolling_term: bool = False         # r·Ω̇: the plate turns under the ball
    frame_terms: bool = False          # Euler, centrifugal and Coriolis accelerations of the rotating plate
    exact_gravity: bool = False        # gravity projected exactly on the tilted plate (couples the axes)
    servo_lag: bool = False            # motors lag their target like the MuJoCo servos (τ = kv/kp)
    sliding: bool = False              # the ball may slide: Coulomb friction μ = mujoco_params.friction
    plate: PlateParams = field(default_factory=lambda: PlateParams(
        bias_x=math.radians(0.5), bias_y=math.radians(-0.3)))
    geometry: Geometry = field(default_factory=Geometry)
    mujoco_params: MujocoParams = field(default_factory=MujocoParams)
    dt_physics: float = 0.0005         # (s): 0.5 ms, required by the elastic contact
    duration: float = 20.0             # (s)
    # Scenario
    start_position: tuple = (0.08, -0.06)   # initial position of the ball (m)
    start_velocity: tuple = (0.0, 0.0)      # (m/s): thrown ball
    start_height: float = 0.0               # (m) above the plate: dropped ball (MuJoCo physics only)
    pushes: tuple = ()                 # impulses (t, Δvx, Δvy)
    trajectory: object = reference     # function t -> (x_ref, y_ref, ax_ref, ay_ref)
    # Motors (if mechanism)
    max_speed: float = 10.0            # (rad/s)
    resolution: float = math.radians(1.8 / 16)   # 200 steps/turn in 1/16 microsteps
    max_acceleration: float = 50.0     # ramps (rad/s²); math.inf: no ramp. Gentle: the plate does not follow
                                       # the noise of the command, which would make the ball hop
    fast_acceleration: float = 400.0   # ramps of the large moves (since B4): a move of 0.34 rad (motor 1 when
                                       # the plate flips from 0 to 10°) takes 0.06 s instead of 0.16 s with
                                       # 50 rad/s² (trapezoidal profile, calculated); peak torque about 0.31 N·m
                                       # in MuJoCo (max 0.4 N·m). To check against a motor datasheet
    fast_move: float = 0.1             # (rad, motor angle) above this gap to its target, a move is large;
                                       # math.inf: gentle ramps only
    fast_speed: float = 0.3            # (m/s) the fast ramps only for a ball measured faster than this, whatever
                                       # the brain (controller/ramps.py): a slow ball would be thrown off the
                                       # plate (0: whatever the speed)
    # Sensor
    sensor: str = "synthetic"          # "synthetic" (Gaussian noise) or "camera" (MuJoCo only)
    sigma: float = 0.001               # synthetic sensor: standard deviation of the noise (m)
    seed: int = 0                      # synthetic sensor: seed of the noise (same seed, same results)
    sigma_height: float = 0.003        # synthetic sensor: noise on the height of the ball (m), like the camera
    latency: int = 1                   # measurement latency, in control periods
    camera: CameraParams = field(default_factory=CameraParams)
    tilt_correction: bool = True
    airborne_height: float = 0.015     # (m) above this measured height, the ball is in the air: the camera takes
                                       # its center on the light ray, at the distance given by its size, and the
                                       # brain aims the bounce (controller/flight.py). math.inf: never in the air
    # Controller
    brain: str = "pid"                 # "pid": the PID brain, with the settings below; or the name of a learned
                                       # policy of brains/ ("imitation"): the AI brain (controller/policy.py)
    control_period: float = 0.02       # 50 Hz
    # Gains of the PID by LQR (controller/lqr.py, Bryson's rule): a position error of lqr_position, an integral
    # of the error of lqr_position·lqr_integral_time and a tilt of lqr_tilt cost as much as one another.
    # Chosen in B3 for the return to the center whatever the calibration error, keeping a margin at 100 ms of
    # latency (stiffer: the loop oscillates there): Kp = 1.89 rad/m, Ki = 1.75 rad/(m·s), Kd = 0.80 rad·s/m
    lqr_position: float = 0.20         # (m)
    lqr_integral_time: float = 0.5     # (s)
    lqr_tilt: float = math.radians(10)
    gains: tuple = None                # (Kp, Ki, Kd) imposed instead of the LQR gains
    theta_max: float = math.radians(10)
    tau_d: float = 0.05                # derivative filter (s); 0: no filter
    feedforward: bool = True
    latency_compensation: bool = True   # predict the current position despite the latency (part of the PID brain)
    bounce_restitution: float = 0.87   # aimed bounce: restitution assumed (ITTF rule 0.85 to 0.91; MuJoCo 0.874)
    bounce_gain: float = 0.5           # aimed bounce: share of the distance to the setpoint covered by each bounce
    wait_for_speed: bool = True        # first measurement: plate kept until the speed of the ball is known
    integration_zone: float = 0.02     # (m) each axis integrates its error only within this distance of the
                                       # setpoint (math.inf: everywhere). It must exceed the offset θb/Kp that
                                       # a calibration error θb leaves without integral: 2 cm corrects up to
                                       # 0.02·Kp rad (2.2° with the LQR gains)
    adapt_model_gain: bool = True      # the model gain c·g of the feedforward and of the latency predictor is
                                       # estimated online (controller/adaptation.py, since B6), from the nominal
                                       # value at every attempt; False: c·g of the hollow ball, fixed

    def pid_gains(self) -> tuple:
        """(Kp, Ki, Kd) of each axis, in rad/m, rad/(m·s) and rad·s/m."""
        if self.gains is not None:
            return self.gains
        return pid_gains(self.plate.coefficient * G, self.lqr_position, self.lqr_integral_time, self.lqr_tilt)


# The frozen reference controller: the PID before the brain was improved (hand-tuned gains Kp = 1 rad/m,
# Ki = 1 rad/(m·s) and the Kd giving a damping ζ = 0.7 to the PD, 2·ζ·√(c·g·Kp) = c·g·Kd; no latency compensation;
# error integrated everywhere; ball never considered in the air; no wait for the speed; model gain fixed), with the
# motors of the first robot (single ramp of 50 rad/s²: the ball never hops, which the simple physics does not model).
# The validation of the simulator (campaign sections 0 and 10) and the regression check use it, so that their
# results do not change when the brain or the robot improves.
_CG = PlateParams().coefficient * G
REFERENCE_GAINS = (1.0, 1.0, 2 * 0.7 * math.sqrt(_CG * 1.0) / _CG)
REFERENCE_CONTROLLER = {"gains": REFERENCE_GAINS, "latency_compensation": False, "integration_zone": math.inf,
                        "airborne_height": math.inf, "wait_for_speed": False, "fast_move": math.inf,
                        "adapt_model_gain": False}
