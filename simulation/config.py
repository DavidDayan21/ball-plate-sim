"""Single configuration of all simulations, and setpoint trajectories.

All physical parameters are initial assumptions (no real reference model):
they can be changed to explore other designs.
"""
from dataclasses import dataclass, field
import math

from controller.kinematics import Geometry

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
    k_actual: float | None = None      # actual inertia of the ball; None: the one the controller assumes


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
def reference(t: float):
    """Reference scenario. 0-4 s: center. 4-8 s: point (5 cm, 5 cm).
    Then: circle of radius 6 cm traveled in 4 s, starting at 45°."""
    if t < 4.0:
        return 0.0, 0.0, 0.0, 0.0
    if t < 8.0:
        return 0.05, 0.05, 0.0, 0.0
    return circle(t - 8.0)


def center(t: float):
    """Ball held at the center."""
    return 0.0, 0.0, 0.0, 0.0


def circle(t: float):
    """Circle of radius 6 cm traveled in 4 s (speed 9.4 cm/s), starting at 45°."""
    R, w = 0.06, 2 * math.pi / 4.0
    phi = w * t + math.pi / 4
    x, y = R * math.cos(phi), R * math.sin(phi)
    return x, y, -w ** 2 * x, -w ** 2 * y


def square(t: float):
    """Square with 10 cm sides at 5 cm/s (2 s per side), starting at the corner (5 cm, 5 cm)."""
    corners = [(0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05)]
    k, s = int(t // 2.0) % 4, (t % 2.0) / 2.0
    (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 0.0, 0.0


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
    max_acceleration: float = 50.0     # ramps (rad/s²); math.inf: no ramp
    # Sensor
    sensor: str = "synthetic"          # "synthetic" (Gaussian noise) or "camera" (MuJoCo only)
    sigma: float = 0.001               # synthetic sensor: standard deviation of the noise (m)
    seed: int = 0                      # synthetic sensor: seed of the noise (same seed, same results)
    sigma_height: float = 0.003        # synthetic sensor: noise on the height of the ball (m), like the camera
    latency: int = 1                   # measurement latency, in control periods
    camera: CameraParams = field(default_factory=CameraParams)
    tilt_correction: bool = True
    airborne_height: float = math.inf  # (m) above this measured height, the ball is in the air: the camera then
                                       # takes its center on the light ray, at the distance given by its size.
                                       # Off for now: the brain reacts worse to the corrected position of a ball
                                       # in the air (campaign section 5); it will come with the aimed bounce (B5)
    # Controller
    control_period: float = 0.02       # 50 Hz
    kp: float = 1.0                    # (rad/m)
    ki: float = 1.0                    # (rad/(m·s))
    zeta: float = 0.7                  # target damping, gives Kd
    theta_max: float = math.radians(10)
    tau_d: float = 0.05                # derivative filter (s); 0: no filter
    feedforward: bool = True
    latency_compensation: bool = True   # predict the current position despite the latency (part of the PID brain)
    integration_zone: float = 0.02     # (m) each axis integrates its error only within this distance of the
                                       # setpoint (math.inf: everywhere). It must exceed the offset θb/Kp that
                                       # a calibration error θb leaves without integral: 2 cm corrects up to
                                       # 0.02 rad = 1.1° with Kp = 1 rad/m (part of the PID brain)

    def kd(self) -> float:
        """Kd giving the target damping ζ: 2·ζ·ω0 = c·g·Kd, with ω0 = √(c·g·Kp)."""
        cg = self.plate.coefficient * G
        return 2 * self.zeta * math.sqrt(cg * self.kp) / cg
