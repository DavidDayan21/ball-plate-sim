"""Synthetic sensor of the position and height of the ball: Gaussian noise + delay (latency).

Also serves as the queue that delays the camera measurements (simulation/camera.py).
"""
from collections import deque
import random


class Sensor:
    """Returns the measured position (x, y) and height of the ball above the plate, with Gaussian noise of
    standard deviation sigma (position) and sigma_height (height), delayed by `latency` control periods.
    Like a real camera, the first measurement arrives `latency` periods after the start: None until then."""

    def __init__(self, sigma: float = 0.001, latency: int = 1, seed: int = 0, sigma_height: float = 0.0):
        self.sigma, self.sigma_height = sigma, sigma_height
        self.latency = latency
        self.rng = random.Random(seed)     # fixed seed: reproducible results
        self.rng_height = random.Random(f"height {seed}")   # its own draws: the position noise does not change
        self.queue = deque()

    def measure(self, x: float, y: float, height: float = 0.0):
        """To be called once per control period. Returns (x_measured, y_measured, height_measured)."""
        self.queue.append((x + self.rng.gauss(0, self.sigma), y + self.rng.gauss(0, self.sigma),
                           height + self.rng_height.gauss(0, self.sigma_height)))
        if len(self.queue) > self.latency:
            return self.queue.popleft()
        return None                           # the first image arrives only after the latency
