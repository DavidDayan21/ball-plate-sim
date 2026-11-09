"""Tracking of the ball in the air: where, when and how fast it will hit the plate (aimed bounce, B5).

In the air, the ball follows a known parabola: constant horizontal velocity, vertical acceleration -g
(air drag neglected: the simulation has none; on a real ping-pong ball at 2.5 m/s, ½·ρ·Cd·A·v² is about
9 % of its weight, calculated with Cd = 0.5). The measurements
arrive in the plate frame and N periods late; each one is brought back to the world frame with the tilt
the camera used for that image, then all the measurements of the current flight are fitted by least squares:
    x(t) = x0 + vx·t,   y(t) = y0 + vy·t,   z(t) + g·t²/2 = z0 + vz·t     (t from the takeoff)
"""
from collections import deque
import math

import numpy as np

from controller.kinematics import rotation


class FlightEstimator:
    """Ballistic estimator of one ball in the air.

    Each control period: observe(measurement), then command(θ) with the tilt just sent to the motors.
    The image of the measurement received now was taken N periods ago, with the tilt commanded one
    period earlier (the camera corrects with the last command): θ_image = command N+1 periods ago.
    At the start, the delay line is not full yet: the k-th measurement is only min(k, N) periods old.
    """

    def __init__(self, plate_height: float, ball_radius: float, period: float, latency: int,
                 airborne_height: float, g: float = 9.81):
        self.plate_height, self.r, self.T, self.N = plate_height, ball_radius, period, latency
        self.airborne_height, self.g = airborne_height, g
        self.commands = deque([(0.0, 0.0)] * (latency + 1), maxlen=latency + 1)
        self.now = -period                     # time of the current control instant: 0 at the first observe
        self.observed = 0
        self.samples = []                     # (t, x, y, z) in the world frame, measurements of the current flight
        self.last = None

    def command(self, theta: tuple) -> None:
        self.commands.append(theta)

    def observe(self, measurement: tuple) -> None:
        """measurement = (x, y, height above the plate), plate frame, N periods late."""
        self.now += self.T
        x, y, h = measurement
        t = self.now - min(self.observed, self.N) * self.T          # time of the image
        self.observed += 1
        if h <= self.airborne_height:                     # ball on the plate: no flight in progress
            self.samples = []
        elif measurement != self.last:                    # a repeated measurement: the ball was not seen
            f = self.fit()
            if f is not None and t > f[0]:                # beyond the predicted impact: a new flight (bounce)
                self.samples = []
            p = np.array(rotation(*self.commands[0], (x, y, self.r + h))) + (0.0, 0.0, self.plate_height)
            self.samples.append((t, *p))
        self.last = measurement

    def fit(self):
        """(time of the impact, center at the impact (x, y, z), velocity at the impact (vx, vy, vz)),
        world frame, times on the clock of observe(); None with fewer than 2 measurements in this flight.
        Impact: the center of the ball reaches the level of a level plate (plate_height + r)."""
        if len(self.samples) < 2:
            return None
        s = np.array(self.samples)
        t0 = s[0, 0]
        t = s[:, 0] - t0
        A = np.column_stack([np.ones_like(t), t])
        (x0, vx), (y0, vy), (z0, vz) = (np.linalg.lstsq(A, b, rcond=None)[0]
                                        for b in (s[:, 1], s[:, 2], s[:, 3] + self.g * t ** 2 / 2))
        # z0 + vz·τ - g·τ²/2 = plate_height + r: the later root
        drop = z0 - self.plate_height - self.r
        tau = (vz + math.sqrt(max(vz * vz + 2 * self.g * drop, 0.0))) / self.g
        return (t0 + tau, (x0 + vx * tau, y0 + vy * tau, self.plate_height + self.r), (vx, vy, vz - self.g * tau))

    def impact(self):
        """(time from now until the impact, center at the impact, velocity at the impact), or None
        if no flight is being tracked."""
        f = self.fit()
        return None if f is None else (f[0] - self.now, f[1], f[2])
