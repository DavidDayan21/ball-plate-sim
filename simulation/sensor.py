"""Synthetic position sensor: Gaussian noise + delay (latency).

Also serves as the queue that delays the camera measurements (simulation/camera.py).
"""
from collections import deque
import random


class Sensor:
    """Returns the measured position with Gaussian noise of standard deviation sigma (m),
    delayed by `latency` control periods."""

    def __init__(self, sigma: float = 0.001, latency: int = 1, seed: int = 0):
        self.sigma = sigma
        self.latency = latency
        self.rng = random.Random(seed)     # fixed seed: reproducible results
        self.queue = deque()

    def measure(self, x: float, y: float):
        """To be called once per control period. Returns (x_measured, y_measured)."""
        self.queue.append((x + self.rng.gauss(0, self.sigma), y + self.rng.gauss(0, self.sigma)))
        if len(self.queue) > self.latency:
            return self.queue.popleft()
        return self.queue[0]                  # at the very beginning: no history yet
