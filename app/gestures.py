"""Mouse gestures on the plate, turned into throws, drops and pushes. Without Qt.

The mouse points at the plate: the ray of the pixel (view3d.Camera.ray) is intersected with the plane of the plate
surface (plate_hit), which gives a point in the frame of the plate. A gesture goes from the point pressed to the point
released: its arrow is the path the ball would cover in ARROW_TIME at the speed given, so speed = length / ARROW_TIME,
up to MAX_SPEED. Beyond the physical limit of the throw, the robot cannot stop the ball: it is allowed, to see it.
"""
import math

import numpy as np

from simulation.metrics import edge_distance, stoppable_speed

ARROW_TIME = 0.2        # (s) a 10 cm arrow throws at 0.5 m/s
MAX_SPEED = 1.5         # (m/s) about twice the physical limit of the reference throw (0.71 m/s)
MIN_SPEED = 0.02        # (m/s) a shorter gesture is a click, not a throw


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
    aim = math.degrees(math.atan2(vy, vx))
    deviation = (aim - angle - 180.0 + 180.0) % 360.0 - 180.0
    return dict(distance=distance, angle_deg=angle, speed=math.hypot(vx, vy), deviation_deg=deviation)


def drop_arguments(point, height: float) -> dict:
    """Arguments of Session.throw for a ball dropped without speed above `point` (m, frame of the plate)."""
    distance = math.hypot(*point)
    angle = math.degrees(math.atan2(point[1], point[0])) if distance > 1e-9 else 0.0
    return dict(distance=distance, angle_deg=angle, height=height)
