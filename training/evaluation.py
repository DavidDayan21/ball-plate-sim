"""Evaluation of a brain on the reference scenario of the campaign, the same for every training script."""
import math

import numpy as np

from simulation.config import Config
from simulation.loop import simulate
from simulation.metrics import max_distance, tracking_error_rms

EVALUATION_SIGMA = 0.0002            # (m) synthetic sensor with the mean error of the camera


def evaluate(brain) -> dict:
    """The brain (function cfg -> brain, see simulation/loop.py) on the reference scenario (MuJoCo robot, sensor
    with the noise of the camera), at 20 and 60 ms of latency: return to the center (max distance between 2 s and
    4 s, objective ≤ 5 mm) and circle error (12 s to 20 s, objective < 2 mm at 60 ms), in mm, NaN if the ball
    falls; tremor of the command on the circle (mean change of θx from one period to the next), in degrees."""
    out = {}
    for latency in (1, 3):
        r = simulate(Config(latency=latency, sigma=EVALUATION_SIGMA), brain=brain)
        ms = 20 * latency
        out[f"return_{ms}ms_mm"] = math.nan if r.fallen and r.t[-1] < 4 else 1000 * max_distance(r, 2, 4)
        out[f"circle_{ms}ms_mm"] = math.nan if r.fallen else 1000 * tracking_error_rms(r, 12, 20)
        circle = [theta for theta, t in zip(r.theta_x, r.t) if t >= 12]
        out[f"tremor_{ms}ms_deg"] = math.nan if r.fallen else float(np.degrees(np.abs(np.diff(circle)).mean()))
    return out
