"""Mouse gestures on the plate, turned into throws, drops and pushes. Without Qt.

The mouse points at the plate: the ray of the pixel (view3d.Camera.ray) is intersected with the plane of the plate
surface (plate_hit), which gives a point in the frame of the plate. A gesture goes from the point pressed to the point
released: its arrow is the path the ball would cover in ARROW_TIME at the speed given, so speed = length / ARROW_TIME,
up to MAX_SPEED. Beyond the physical limit of the throw, the robot cannot stop the ball: it is allowed, to see it.

A drop with a speed: the ball is released above the point pressed, with the horizontal speed of the arrow, and falls
along a parabola (the model has no air) until it touches the plate (drop_flight).
"""
import math

import numpy as np

from simulation.config import G
from simulation.metrics import edge_distance, stoppable_speed

ARROW_TIME = 0.2        # (s) a 10 cm arrow throws at 0.5 m/s
MAX_SPEED = 1.5         # (m/s) about twice the physical limit of the reference throw (0.71 m/s)
MIN_SPEED = 0.02        # (m/s) a shorter gesture is a click, not a throw
FLIGHT_POINTS = 24      # points of the parabola drawn for a drop


def plate_hit(eye, direction, plate_pos, plate_mat):
    """Where the ray eye + s·direction (s > 0) meets the plane of the plate surface: (x, y) in the frame of the plate
    (m), or None if it misses (parallel, or behind the eye). plate_pos, plate_mat: origin (on the surface) and
    rotation matrix of the plate (MuJoCo xpos, xmat)."""
    R = np.asarray(plate_mat, dtype=float).reshape(3, 3)
    normal = R[:, 2]
    along = float(np.dot(direction, normal))
    if abs(along) < 1e-9:
        return None
    s = float(np.dot(np.asarray(plate_pos) - eye, normal)) / along
    if s <= 0:
        return None
    local = R.T @ (np.asarray(eye) + s * np.asarray(direction) - np.asarray(plate_pos))
    return float(local[0]), float(local[1])


def velocity(start, end) -> tuple:
    """(vx, vy) of the gesture from start to end (m, frame of the plate): length / ARROW_TIME, up to MAX_SPEED."""
    vx, vy = (end[0] - start[0]) / ARROW_TIME, (end[1] - start[1]) / ARROW_TIME
    speed = math.hypot(vx, vy)
    if speed > MAX_SPEED:
        vx, vy = vx * MAX_SPEED / speed, vy * MAX_SPEED / speed
    return vx, vy


def throw_limit(start, vx: float, vy: float, plate_radius: float, theta_max: float, model_gain: float) -> float:
    """Physical limit (m/s) of a throw from `start` in the direction of (vx, vy): the speed that the plate, tilted by
    theta_max against the ball, can just stop before the edge (simulation.metrics.stoppable_speed)."""
    d = edge_distance(start[0], start[1], math.atan2(vy, vx), plate_radius)
    return stoppable_speed(max(d, 0.0), theta_max, model_gain)


def throw_arguments(start, vx: float, vy: float) -> dict:
    """Arguments of Session.throw for a ball starting at `start` (m, frame of the plate) with velocity (vx, vy):
    distance and angle of the start from the center, speed, deviation of the velocity from the center."""
    distance = math.hypot(*start)
    angle = math.degrees(math.atan2(start[1], start[0])) if distance > 1e-9 else 0.0
    speed = math.hypot(vx, vy)
    aim = math.degrees(math.atan2(vy, vx))
    deviation = (aim - angle - 180.0 + 180.0) % 360.0 - 180.0 if speed > 0 else 0.0
    return dict(distance=distance, angle_deg=angle, speed=speed, deviation_deg=deviation)


def drop_arguments(point, height: float, vx: float = 0.0, vy: float = 0.0) -> dict:
    """Arguments of Session.throw for a ball dropped from `height` (m, its bottom above the plate) above `point` (m,
    frame of the plate), with the horizontal velocity (vx, vy) (m/s, frame of the plate; none by default)."""
    return dict(throw_arguments(point, vx, vy), height=height, action="drop")


def drop_flight(start, vx: float, vy: float, height: float, plate_mat=None) -> tuple:
    """Fall of a ball released with its bottom at `height` (m) above the point `start` of the plate, with the velocity
    (vx, vy) along the plate (frame of the plate), until it touches the plane of the plate, assumed to stay where it
    is (plate_mat: its rotation in the world, horizontal if None). The model has no air: the ball falls along a
    parabola, under the gravity G of the world seen from the plate, g = R^T·(0, 0, -G).
    Bottom of the ball at the time t: start + v·t + g·t²/2 (height + g_z·t²/2 above the plate), until
    t = √(2·height / -g_z).
    Returns (landing (x, y), angle of the arrival from the normal of the plate (°), path: FLIGHT_POINTS points
    (x, y, height of the bottom of the ball) from the release to the landing)."""
    R = np.eye(3) if plate_mat is None else np.asarray(plate_mat, dtype=float).reshape(3, 3)
    gx, gy, gz = R.T @ np.array([0.0, 0.0, -G])
    t_land = math.sqrt(2.0 * max(height, 0.0) / -gz)
    path = [(start[0] + vx * t + gx * t * t / 2, start[1] + vy * t + gy * t * t / 2, height + gz * t * t / 2)
            for t in np.linspace(0.0, t_land, FLIGHT_POINTS)]
    along = math.hypot(vx + gx * t_land, vy + gy * t_land)
    return path[-1][:2], math.degrees(math.atan2(along, -gz * t_land)), path
