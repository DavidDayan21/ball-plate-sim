"""Simple physics: our own equations, without an external physics engine.

Basic model (the default), decoupled, small angles, integrated with semi-implicit Euler:
    ẍ = c·g·sin(θx + bx)        ÿ = c·g·sin(θy + by)        c = 1 / (1 + k)

Full equation of a ball rolling without slipping on a plate that rotates about the center O of its
surface (derived in the plate frame, and checked against MuJoCo: tests/test_physics.py):
    (1 + k)·(ẍ + r·Ω̇2) = g1 + Ω̇3·y − [Ω×(Ω×p)]1 − 2·[Ω×ṗ]1
    (1 + k)·(ÿ − r·Ω̇1) = g2 − Ω̇3·x − [Ω×(Ω×p)]2 − 2·[Ω×ṗ]2
with p = (x, y, r) the center of the ball, ṗ = (ẋ, ẏ, 0), Ω = (−θ̇y, θ̇x·cos θy, θ̇x·sin θy) the angular
velocity of the plate (rotation R = Ry(θx)·Rx(−θy)) and (g1, g2) = (g·sin θx, g·cos θx·sin θy) the
gravity along the plate. If O itself accelerates (aO), aO is subtracted from (g1, g2): not modeled
here, O is fixed. In 1D: ẍ = (g·sin θ + x·θ̇²)/(1 + k) − r·θ̈.

The additions to the basic model can be switched on one by one (Config, all off by default):
    rolling_term     r·Ω̇: the plate turns under the ball (in 1D, the ball moves by −r·Δθ)
    frame_terms      Euler (Ω̇3), centrifugal and Coriolis accelerations of the rotating plate
    exact_gravity    gravity projected exactly on the tilted plate (cos θx couples the axes)
    servo_lag        the motors (or the two angles of the angle-driven plate) follow their target with a
                     first-order lag τ = kv/kp, like the position servos of the MuJoCo model

Assumptions: rolling without slipping, permanent contact, spin of the ball about the normal neglected
(third-order terms). With mechanism=True, the angles go through the 3 motors (simulation/motor.py)
and the actual tilt is given by the forward kinematics; with mechanism=False, the two angles of the
plate follow their target with the same profile as the motors, without steps.

The basic model serves as the theoretical reference: its results can be derived by hand
(see tests/test_controller.py).
"""
import math

from controller.kinematics import forward_kinematics, inverse_kinematics
from simulation.config import G, Config, PlateParams
from simulation.motor import motors


def acceleration(theta_x: float, theta_y: float, p: PlateParams):
    """Acceleration (ax, ay) of the ball (m/s²) for the commanded angles (θx, θy)."""
    cg = p.coefficient * G
    return cg * math.sin(theta_x + p.bias_x), cg * math.sin(theta_y + p.bias_y)


def semi_implicit_step(x, y, vx, vy, theta_x, theta_y, dt, p: PlateParams):
    """One semi-implicit Euler step: the new velocity is used to compute the new position."""
    ax, ay = acceleration(theta_x, theta_y, p)
    vx += ax * dt
    vy += ay * dt
    return x + vx * dt, y + vy * dt, vx, vy


def rotating_plate_acceleration(p, v, omega, domega, gravity, k: float, frame: bool = True):
    """(ẍ + r·Ω̇2, ÿ − r·Ω̇1) given by the full equation (see the module docstring), in the plate frame.

    p = (x, y, r), v = (ẋ, ẏ), omega = Ω, domega = Ω̇, gravity = (g1, g2), minus aO if O accelerates.
    frame=False: gravity only (no Euler, centrifugal or Coriolis term).
    """
    g1, g2 = gravity
    if frame:
        x, y, r = p
        o1, o2, o3 = omega
        dot = o1 * x + o2 * y + o3 * r                  # Ω·p
        norm2 = o1 * o1 + o2 * o2 + o3 * o3             # |Ω|²; −Ω×(Ω×p) = |Ω|²·p − (Ω·p)·Ω
        g1 += domega[2] * y + norm2 * x - dot * o1 + 2 * o3 * v[1]     # −2·Ω×ṗ = (2·Ω3·ẏ, −2·Ω3·ẋ)
        g2 += -domega[2] * x + norm2 * y - dot * o2 - 2 * o3 * v[0]
    return g1 / (1 + k), g2 / (1 + k)


class SimplePhysics:
    def __init__(self, cfg: Config):
        self.p, self.g = cfg.plate, cfg.geometry
        self.mechanism = cfg.mechanism
        self.x, self.y = cfg.start_position
        self.vx, self.vy = cfg.start_velocity
        self.unreachable_poses = 0
        # Additions to the basic model
        self.rolling, self.frame, self.exact_gravity = cfg.rolling_term, cfg.frame_terms, cfg.exact_gravity
        self.extended = self.rolling or self.frame or self.exact_gravity
        pm = cfg.mujoco_params
        self.servo_tau = pm.kv_servo / pm.kp_servo if cfg.servo_lag else 0.0
        if cfg.start_height:
            raise ValueError("The simple physics keeps the ball on the plate: a dropped ball needs MuJoCo")
        self.k, self.r = cfg.plate.k_inertia, cfg.plate.ball_radius
        self.previous_tilt = (0.0, 0.0)
        self.omega = (0.0, 0.0, 0.0)
        if self.mechanism:
            self.alphas = inverse_kinematics(self.g, 0.0, 0.0)
            self.motors = motors(cfg, self.alphas, cfg.resolution)
            self.pose = (0.0, 0.0, self.g.height)
        else:
            self.tilt = (0.0, 0.0)              # the two angles of the plate, driven like motors
            self.motors = motors(cfg, self.tilt, 0.0)

    def ball(self):
        """Position (x, y) of the ball in the plate frame (m)."""
        return self.x, self.y

    def fallen(self) -> bool:
        return math.hypot(self.x, self.y) > self.p.plate_radius

    def command(self, theta_x: float, theta_y: float, fast: bool = True) -> None:
        """fast: the motors may use their fast ramp for the large moves (simulation/motor.py)."""
        for m in self.motors:
            m.fast_allowed = fast
        if not self.mechanism:
            for m, a in zip(self.motors, (theta_x, theta_y)):
                m.request(a)
        else:
            try:
                for m, a in zip(self.motors, inverse_kinematics(self.g, theta_x, theta_y)):
                    m.request(a)
            except ValueError:
                self.unreachable_poses += 1     # impossible pose: the motors keep their target

    def actual_tilt(self):
        return (self.pose[0], self.pose[1]) if self.mechanism else self.tilt

    def motor_angles(self):
        return self.alphas if self.mechanism else ()

    def ball_velocity(self):
        return self.vx, self.vy

    def ball_height(self) -> float:
        return 0.0                              # planar model: the ball stays in contact

    def push(self, dvx: float, dvy: float) -> None:
        self.vx += dvx
        self.vy += dvy

    def place_ball(self, x, y, z=0.0, vx=0.0, vy=0.0, vz=0.0) -> None:
        """Teleports the ball (plate frame). The planar model ignores the height z and vz."""
        self.x, self.y, self.vx, self.vy = x, y, vx, vy

    def step(self, dt: float) -> None:
        angles = self.alphas if self.mechanism else self.tilt
        new_angles = tuple(m.advance(dt) for m in self.motors)
        if self.servo_tau:                      # first-order lag, discretized exactly
            gain = 1 - math.exp(-dt / self.servo_tau)
            new_angles = tuple(a + (t - a) * gain for a, t in zip(angles, new_angles))
        if new_angles != angles:                # the pose only changes if a motor has moved
            if self.mechanism:
                self.alphas = new_angles
                self.pose = forward_kinematics(self.g, self.alphas, initial_pose=self.pose)
            else:
                self.tilt = new_angles
        tx, ty = self.actual_tilt()
        if not self.extended:
            self.x, self.y, self.vx, self.vy = semi_implicit_step(self.x, self.y, self.vx, self.vy,
                                                                  tx, ty, dt, self.p)
            return
        # Full equation: Ω from the tilts by finite differences, angles of the plate including the bias
        bx, by = tx + self.p.bias_x, ty + self.p.bias_y
        dtx, dty = (tx - self.previous_tilt[0]) / dt, (ty - self.previous_tilt[1]) / dt
        self.previous_tilt = (tx, ty)
        omega = (-dty, dtx * math.cos(by), dtx * math.sin(by))
        domega = tuple((a - b) / dt for a, b in zip(omega, self.omega))
        previous, self.omega = self.omega, omega
        g1 = G * math.sin(bx)
        g2 = G * math.cos(bx) * math.sin(by) if self.exact_gravity else G * math.sin(by)
        a1, a2 = rotating_plate_acceleration((self.x, self.y, self.r), (self.vx, self.vy), omega, domega,
                                             (g1, g2), self.k, self.frame)
        self.vx += a1 * dt
        self.vy += a2 * dt
        if self.rolling:                        # d/dt(ẋ + r·Ω2) = a1 and d/dt(ẏ − r·Ω1) = a2
            self.vx -= self.r * (omega[1] - previous[1])
            self.vy += self.r * (omega[0] - previous[0])
        self.x += self.vx * dt
        self.y += self.vy * dt
