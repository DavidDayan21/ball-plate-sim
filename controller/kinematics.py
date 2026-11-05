"""Kinematics of the 3-arm mechanism.

Mechanism: 3 motors placed at 120° around the center. Each motor rotates an arm
of length l1 in the vertical radial plane. The tip of the arm is connected to the plate by a
rod of length l2, with a ball joint at each end.

Frame: origin at the level of the motor axes, on the central axis; z pointing up.
Plate pose: center at (0, 0, h), orientation given by the tilts (θx, θy) of the
controller, with the rotation R = Ry(θx)·Rx(-θy): θx > 0 lowers the +x side, θy > 0 the +y side.

Arm angle α: 0 = horizontal arm pointing outward, > 0 upward.

- inverse_kinematics: desired pose -> angles of the 3 motors (exact formula). Runs on the robot.
- forward_kinematics: motor angles -> actual pose (Newton's method). Used by the simulator.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    """Initial geometry (an assumption, to be matched to a real model if the robot is built).

    Chosen so that at rest the arm is horizontal and the rod vertical:
    attachment_radius = base_radius + arm_length and height = rod_length.
    """
    base_radius: float = 0.05           # center -> axis of each motor (m)
    arm_length: float = 0.05        # l1 (m)
    rod_length: float = 0.12   # l2 (m)
    attachment_radius: float = 0.10        # plate center -> ball joint of the rod (m)
    height: float = 0.12              # nominal height of the plate center (m)
    arm_angles: tuple = (0.0, 120.0, 240.0)   # angular position of the 3 arms (°)

    def direction(self, i: int):
        """Radial unit vector (ux, uy) of arm i."""
        phi = math.radians(self.arm_angles[i])
        return math.cos(phi), math.sin(phi)


def rotation(theta_x: float, theta_y: float, v):
    """Applies R = Ry(θx)·Rx(-θy) to the vector v = (x, y, z)."""
    x, y, z = v
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    y1, z1 = cy * y + sy * z, -sy * y + cy * z          # Rx(-θy)
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    return cx * x + sx * z1, y1, -sx * x + cx * z1      # Ry(θx)


def plate_points(g: Geometry, theta_x: float, theta_y: float, h: float):
    """Positions of the 3 ball joints on the plate side."""
    points = []
    for i in range(3):
        ux, uy = g.direction(i)
        px, py, pz = rotation(theta_x, theta_y, (g.attachment_radius * ux, g.attachment_radius * uy, 0.0))
        points.append((px, py, pz + h))
    return points


def arm_tip(g: Geometry, i: int, alpha: float):
    """Position of the ball joint on the arm side, for arm i at angle α."""
    ux, uy = g.direction(i)
    r = g.base_radius + g.arm_length * math.cos(alpha)
    return r * ux, r * uy, g.arm_length * math.sin(alpha)


def inverse_kinematics(g: Geometry, theta_x: float, theta_y: float, h: float = None):
    """Motor angles (α1, α2, α3) that give the desired pose.

    For each arm: let D = P - B (plate ball joint minus motor axis), and du and dz its
    radial and vertical components. The constraint |P - A(α)| = l2 gives
        du·cos α + dz·sin α = K,   with K = (|D|² + l1² - l2²) / (2·l1)
    i.e. ρ·cos(α - γ) = K, with ρ = √(du² + dz²) and γ = atan2(dz, du):
        α = γ - arccos(K / ρ)       ("arm pointing outward" branch)
    Raises ValueError if the pose is out of reach (|K| > ρ).
    """
    h = g.height if h is None else h
    l1, l2 = g.arm_length, g.rod_length
    alphas = []
    for i, (px, py, pz) in enumerate(plate_points(g, theta_x, theta_y, h)):
        ux, uy = g.direction(i)
        dx, dy, dz = px - g.base_radius * ux, py - g.base_radius * uy, pz
        du = dx * ux + dy * uy
        K = (dx * dx + dy * dy + dz * dz + l1 * l1 - l2 * l2) / (2 * l1)
        rho = math.hypot(du, dz)
        if abs(K) > rho:
            raise ValueError(f"Pose out of reach of the mechanism (arm {i + 1})")
        alphas.append(math.atan2(dz, du) - math.acos(K / rho))
    return tuple(alphas)


def _residuals(g: Geometry, alphas, q):
    """|P_i - A_i|² - l2² for the 3 arms (zero when the pose q is consistent with the angles)."""
    theta_x, theta_y, h = q
    res = []
    for i, P in enumerate(plate_points(g, theta_x, theta_y, h)):
        A = arm_tip(g, i, alphas[i])
        res.append(sum((p - a) ** 2 for p, a in zip(P, A)) - g.rod_length ** 2)
    return res


def _solve_3x3(J, b):
    """Solves J·x = b with Cramer's rule."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(J)
    solution = []
    for k in range(3):
        m = [row[:] for row in J]
        for r in range(3):
            m[r][k] = b[r]
        solution.append(det(m) / d)
    return solution


def forward_kinematics(g: Geometry, alphas, initial_pose=None, tol: float = 1e-12, iterations: int = 30):
    """Plate pose (θx, θy, h) for given motor angles (Newton's method).

    initial_pose: starting pose (the previous one, in simulation: converges in 1 or 2 iterations).
    """
    q = list(initial_pose) if initial_pose is not None else [0.0, 0.0, g.height]
    eps = 1e-7
    for _ in range(iterations):
        r = _residuals(g, alphas, q)
        J = [[0.0] * 3 for _ in range(3)]
        for k in range(3):
            q2 = q[:]
            q2[k] += eps
            r2 = _residuals(g, alphas, q2)
            for i in range(3):
                J[i][k] = (r2[i] - r[i]) / eps
        dq = _solve_3x3(J, [-v for v in r])
        q = [a + b for a, b in zip(q, dq)]
        if max(abs(v) for v in dq) < tol:
            return tuple(q)
    raise RuntimeError("Forward kinematics did not converge")
