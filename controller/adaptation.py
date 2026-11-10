"""Adaptive control (B6): online estimation of the model gain c·g.

The controller assumes ẍ = c·g·θ: the feedforward θ_ff = a_ref / (c·g) and the latency predictor use c·g.
c = 1/(1 + k) depends on the inertia of the ball (k = 2/3 hollow, 2/5 solid): for a solid ball, c·g is 19 % higher
than assumed, the feedforward overshoots and the circle error is more than three times larger (campaign, section 11).

Model, on each axis, at the control period T:
    (x(k) - 2·x(k-1) + x(k-2)) / T² = G·(θ + θb)
x: the measured positions, N periods late; the acceleration over the two periods around x(k-1) is driven by the
commands sent N + 1 and N + 2 periods earlier (their mean is θ). θb, the calibration error of the plate, is
unknown: the offset b = G·θb is estimated with G.

Both sides go through the same low-pass filter (two first-order stages at 3 rad/s): the second difference
amplifies the measurement noise (σ·√6/T², 6 m/s² for σ = 1 mm), and the high frequencies carry what the model
ignores (the plate turning under the ball, the ramps of the motors). The model is linear: the filtered signals obey
the same relation.

Unknowns (G, bx, by): recursive least squares with exponential forgetting, starting from the nominal gain at every
attempt (nothing is kept from one attempt to the next). Prior: G0 ± 30 % and offsets ± 0.02·G0 (about 1° of
calibration error), for a residual of 0.05 m/s² (measured on the circle in MuJoCo: 0.003 m/s² with the camera,
0.011 m/s² with the synthetic sensor; the filtered residual is correlated over about 15 samples, hence the margin).
The covariance is kept below the prior: without excitation, the estimate stays where it is instead of winding up.

The equations are used only where the model holds (measured on the prototype, MuJoCo, true positions and true
tilts of the plate, reference scenario):
- ball on the plate, command not saturated;
- ball tracking its setpoint within 1 cm for at least 1 s: during a transient (step of the setpoint, throw), the
  plate lags its command by up to 10° (motor ramps); once tracking, by 0.3° at most (95 %);
- ball rolling at 3 cm/s at least: near rest, the fit gives G = 2 to 4 instead of 5.9 (hollow) and 7.0 (solid).
Elsewhere, G keeps its last value. It stays between 0.5 and 2 times the nominal gain (safety bound).
"""
from collections import deque
import math

import numpy as np


class ModelGainEstimator:
    """Online estimate of G = c·g (shared by both axes), from the measured positions and the commands sent.

    Call update() with each measurement, before the command is computed, then command() with the command sent."""

    def __init__(self, nominal_gain: float, period: float, latency: int, theta_max: float,
                 cutoff: float = 3.0, memory: float = 1.5, prior: float = 0.3, offset_prior: float = 0.02,
                 residual: float = 0.05, min_speed: float = 0.03, max_error: float = 0.01, calm_time: float = 1.0):
        """nominal_gain: c·g assumed (m/s² per rad). period (s), latency (periods) of the measurement.
        cutoff (rad/s): low-pass filter. memory (s): time constant of the forgetting. prior, offset_prior: standard
        deviations of the prior, relative to nominal_gain. residual (m/s²): assumed error of the filtered equation.
        min_speed (m/s), max_error (m), calm_time (s): the equations are used only for a ball rolling faster than
        min_speed, within max_error of its setpoint for at least calm_time."""
        self.T, self.N, self.theta_max, self.g0 = period, latency, theta_max, nominal_gain
        self.a = math.exp(-cutoff * period)
        self.forgetting = math.exp(-period / memory)
        self.theta = np.array([nominal_gain, 0.0, 0.0])                    # (G, bx, by)
        self.P_max = np.diag([(prior * nominal_gain) ** 2, (offset_prior * nominal_gain) ** 2,
                              (offset_prior * nominal_gain) ** 2]) / residual ** 2
        self.P = self.P_max.copy()
        self.min_speed, self.max_error = min_speed, max_error
        self.calm_needed = round(calm_time / period)
        self.calm = 0                                                      # periods since the last disturbance
        self.positions = deque(maxlen=3)                                   # x(k-2), x(k-1), x(k)
        self.commands = deque([(0.0, 0.0)] * (latency + 2), maxlen=latency + 2)   # u(k-N-2) ... u(k-1)
        self.setpoints = deque(maxlen=latency + 1)                         # setpoint(k-N) ... setpoint(k)
        self.filters = {}                                                  # two-stage low-pass, per signal

    def _low_pass(self, key: str, value: float) -> float:
        s = self.filters.setdefault(key, [0.0, 0.0])
        s[0] = self.a * s[0] + (1 - self.a) * value
        s[1] = self.a * s[1] + (1 - self.a) * s[0]
        return s[1]

    def update(self, measured, setpoint, airborne: bool) -> None:
        """measured: (x, y) as received (N periods late); setpoint: (x, y) of this period."""
        self.positions.append(tuple(measured))
        self.setpoints.append(tuple(setpoint))
        # Calm: on the plate, command not saturated, within max_error of the setpoint of the time of the measurement
        late_setpoint = self.setpoints[0]
        error = math.hypot(measured[0] - late_setpoint[0], measured[1] - late_setpoint[1])
        saturated = max(abs(c) for c in self.commands[-1]) > 0.98 * self.theta_max
        self.calm = 0 if airborne or saturated or error > self.max_error else self.calm + 1
        if len(self.positions) < 3:
            return
        p = self.positions
        filtered = []
        for i in range(2):
            acceleration = (p[2][i] - 2 * p[1][i] + p[0][i]) / self.T ** 2
            command = (self.commands[0][i] + self.commands[1][i]) / 2       # u(k-N-2), u(k-N-1)
            filtered.append((self._low_pass(f"a{i}", acceleration), self._low_pass(f"u{i}", command),
                             self._low_pass(f"v{i}", (p[2][i] - p[1][i]) / self.T)))
        one = self._low_pass("one", 1.0)
        speed = math.hypot(filtered[0][2], filtered[1][2])
        if self.calm < self.calm_needed or speed < self.min_speed:
            return
        for i, (y, u, _) in enumerate(filtered):
            phi = np.array([u, one if i == 0 else 0.0, one if i == 1 else 0.0])
            forgetting = self.forgetting if i == 0 else 1.0                # once per period, for both equations
            P_phi = self.P @ phi
            k = P_phi / (forgetting + phi @ P_phi)
            self.theta = self.theta + k * (y - phi @ self.theta)
            self.P = (self.P - np.outer(k, P_phi)) / forgetting
            for j in range(3):                                             # no windup beyond the prior
                if self.P[j, j] > self.P_max[j, j]:
                    s = math.sqrt(self.P_max[j, j] / self.P[j, j])
                    self.P[j, :] *= s
                    self.P[:, j] *= s
        self.theta[0] = min(max(self.theta[0], 0.5 * self.g0), 2.0 * self.g0)

    def command(self, theta) -> None:
        """(θx, θy) sent this period."""
        self.commands.append(tuple(theta))

    @property
    def gain(self) -> float:
        return float(self.theta[0])
