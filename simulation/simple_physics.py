"""Simple physics: our own equations, without an external physics engine.

Ball on a plate, decoupled model (small angles), integrated with semi-implicit Euler:
    ẍ = c·g·sin(θx + bx)        ÿ = c·g·sin(θy + by)        c = 1 / (1 + k)

Assumptions: rolling without slipping, permanent contact, coupling between axes and effects of the
plate rotation neglected. With mechanism=True, the angles go through the 3 motors
(simulation/motor.py) and the actual tilt is given by the forward kinematics.

Serves as the theoretical reference: its results can be derived by hand (see tests/test_controller.py).
"""
import math

from controller.kinematics import forward_kinematics, inverse_kinematics
from simulation.config import G, Config, PlateParams
from simulation.motor import Motor


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


class SimplePhysics:
    def __init__(self, cfg: Config):
        self.p, self.g = cfg.plate, cfg.geometry
        self.mechanism = cfg.mechanism
        self.x, self.y = cfg.start_position
        self.vx, self.vy = cfg.start_velocity
        self.commanded_tilt = (0.0, 0.0)
        self.unreachable_poses = 0
        if self.mechanism:
            self.alphas = inverse_kinematics(self.g, 0.0, 0.0)
            self.motors = [Motor(a, cfg.max_speed, cfg.resolution, max_acceleration=cfg.max_acceleration)
                            for a in self.alphas]
            self.pose = (0.0, 0.0, self.g.height)

    def ball(self):
        """Position (x, y) of the ball in the plate frame (m)."""
        return self.x, self.y

    def fallen(self) -> bool:
        return math.hypot(self.x, self.y) > self.p.plate_radius

    def command(self, theta_x: float, theta_y: float) -> None:
        self.commanded_tilt = (theta_x, theta_y)
        if self.mechanism:
            try:
                for m, a in zip(self.motors, inverse_kinematics(self.g, theta_x, theta_y)):
                    m.request(a)
            except ValueError:
                self.unreachable_poses += 1     # impossible pose: the motors keep their target

    def actual_tilt(self):
        return (self.pose[0], self.pose[1]) if self.mechanism else self.commanded_tilt

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
        if self.mechanism:
            new_angles = tuple(m.advance(dt) for m in self.motors)
            if new_angles != self.alphas:          # the pose only changes if a motor has moved
                self.alphas = new_angles
                self.pose = forward_kinematics(self.g, self.alphas, initial_pose=self.pose)
        tx, ty = self.actual_tilt()
        self.x, self.y, self.vx, self.vy = semi_implicit_step(self.x, self.y, self.vx, self.vy,
                                                              tx, ty, dt, self.p)
