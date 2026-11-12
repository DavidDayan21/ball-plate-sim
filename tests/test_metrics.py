"""Return-to-center metrics, on hand-made histories."""
import math

import pytest

from simulation.loop import Results
from simulation.metrics import arrival_time, max_distance, overshoot, stoppable_speed


def history(points, dt: float = 0.5) -> Results:
    """Ball at the given (x, y) positions every dt, setpoint at the center."""
    r = Results()
    for i, (x, y) in enumerate(points):
        r.t.append(i * dt)
        r.x.append(x)
        r.y.append(y)
        r.x_ref.append(0.0)
        r.y_ref.append(0.0)
    return r


#         t = 0          0.5          1.0           1.5          2.0           2.5               3.0
APPROACH = [(0.1, 0.0), (0.06, 0.0), (0.004, 0.0), (0.03, 0.0), (0.0, 0.01), (0.003, 0.004), (0.002, 0.0)]


def test_arrival_overshoot_and_distance_after_2_s():
    r = history(APPROACH)
    assert arrival_time(r, 4) == 1.0                         # first distance below 5 % of 10 cm
    assert overshoot(r, 4) == pytest.approx(0.03)            # largest distance after the arrival
    assert max_distance(r, 2, 4) == pytest.approx(0.01)      # distance in 2D: hypot(0, 0.01)
    assert max_distance(r, 2.5, 4) == pytest.approx(0.005)   # hypot(0.003, 0.004)


def test_no_arrival():
    r = history([(0.1, 0.0), (0.08, 0.0), (0.06, 0.0)])
    assert math.isnan(arrival_time(r, 4)) and math.isnan(overshoot(r, 4))


def test_distance_after_2_s_is_continuous():
    # A late peak of 4.9 mm or 5.1 mm: the objective flips, but the metric only moves by 0.2 mm
    # (the former settling time jumped by more than a second in this situation)
    low = history(APPROACH[:5] + [(0.0049, 0.0), (0.0, 0.0)])
    high = history(APPROACH[:5] + [(0.0051, 0.0), (0.0, 0.0)])
    assert max_distance(high, 2.5, 4) - max_distance(low, 2.5, 4) == pytest.approx(0.0002)
    assert arrival_time(low, 4) == arrival_time(high, 4) == 1.0


def test_stoppable_speed():
    # Ball thrown from 10 cm before the center toward the opposite edge, 25 cm away, plate at 10°, hollow ball
    # (c·g = 3/5·g): 0.71 m/s (campaign, section 2). Braked at c·g·sin θ, it stops after v²/(2·a) = 25 cm
    a = 0.6 * 9.81 * math.sin(math.radians(10))
    v = stoppable_speed(0.25, math.radians(10), 0.6 * 9.81)
    assert v == pytest.approx(0.715, abs=0.001) and v ** 2 / (2 * a) == pytest.approx(0.25)
