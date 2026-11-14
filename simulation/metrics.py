"""Performance metrics computed on a history (simulation/loop.py: Results), and the physical limit of a throw."""
import math


def indices(r, t0: float, t1: float):
    return [i for i, t in enumerate(r.t) if t0 <= t < t1]


def tracking_error_rms(r, t0: float, t1: float) -> float:
    """Root-mean-square distance between the ball and the setpoint over [t0, t1[ (m)."""
    idx = indices(r, t0, t1)
    return math.sqrt(sum((r.x[i] - r.x_ref[i]) ** 2 + (r.y[i] - r.y_ref[i]) ** 2 for i in idx) / len(idx))


def distance(r, i: int) -> float:
    """Distance between the ball and the setpoint at sample i (m)."""
    return math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i])


# Return to the center. A settling time (the last time the distance exceeds a threshold) is not used:
# when the response oscillates, it jumps as soon as a peak crosses the threshold (a 4.9 mm or a 5.1 mm
# peak changed it by more than a second). The metrics below vary continuously with the response.

def max_distance(r, t0: float, t1: float) -> float:
    """Maximum distance between the ball and the setpoint over [t0, t1[ (m).
    Objective "back to the center in less than 2 s": max_distance(r, 2, 4) ≤ 5 mm."""
    return max(distance(r, i) for i in indices(r, t0, t1))


def arrival_time(r, t1: float, fraction: float = 0.05) -> float:
    """First time the distance falls below `fraction` of the initial distance, before t1 (s).
    NaN if it never does."""
    d0 = distance(r, 0)
    return next((r.t[i] for i in indices(r, 0.0, t1) if distance(r, i) < fraction * d0), math.nan)


def overshoot(r, t1: float, fraction: float = 0.05) -> float:
    """Maximum distance after the arrival (see arrival_time), before t1 (m). NaN without arrival."""
    t_arrival = arrival_time(r, t1, fraction)
    if math.isnan(t_arrival):
        return math.nan
    return max_distance(r, t_arrival, t1)


def edge_distance(x: float, y: float, direction: float, radius: float) -> float:
    """Distance from (x, y) to the edge of a plate of this radius, along the angle `direction`."""
    ux, uy = math.cos(direction), math.sin(direction)
    along = x * ux + y * uy
    return -along + math.sqrt(along ** 2 + radius ** 2 - x * x - y * y)


def stoppable_speed(distance: float, theta_max: float, model_gain: float) -> float:
    """Highest speed (m/s) of a ball that the plate can stop over `distance`: tilted by theta_max against it, the
    plate brakes it by a = c·g·sin θmax (model_gain = c·g), hence v = √(2·a·d). The physical limit of any brain."""
    return math.sqrt(2 * model_gain * math.sin(theta_max) * distance)


def theta_std(r, t0: float, t1: float) -> float:
    """Standard deviation of θx over [t0, t1[ (rad): command jitter caused by the noise."""
    vals = [r.theta_x[i] for i in indices(r, t0, t1)]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
