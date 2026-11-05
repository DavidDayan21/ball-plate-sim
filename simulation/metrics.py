"""Performance metrics computed on a history (simulation/loop.py: Results)."""
import math


def indices(r, t0: float, t1: float):
    return [i for i, t in enumerate(r.t) if t0 <= t < t1]


def tracking_error_rms(r, t0: float, t1: float) -> float:
    """Root-mean-square distance between the ball and the setpoint over [t0, t1[ (m)."""
    idx = indices(r, t0, t1)
    return math.sqrt(sum((r.x[i] - r.x_ref[i]) ** 2 + (r.y[i] - r.y_ref[i]) ** 2 for i in idx) / len(idx))


def settling_time_5(r, t1: float) -> float:
    """Settling time to within 5 % of the initial distance to the setpoint (before t1)."""
    d0 = math.hypot(r.x[0] - r.x_ref[0], r.y[0] - r.y_ref[0])
    for i in reversed(indices(r, 0.0, t1)):
        if math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) > 0.05 * d0:
            return r.t[i]
    return 0.0


def theta_std(r, t0: float, t1: float) -> float:
    """Standard deviation of θx over [t0, t1[ (rad): command jitter caused by the noise."""
    vals = [r.theta_x[i] for i in indices(r, t0, t1)]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
