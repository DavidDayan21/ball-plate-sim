"""2D plate controller: one PID per axis + feedforward
+ compensation of the measurement latency by prediction (optional, disabled by default)
+ conditional integration (optional, disabled by default).

Independent of the simulator: it receives the setpoint, the measurement and the desired acceleration,
and returns the two tilts (θx, θy) to apply to the plate.
"""
from collections import deque
import math

from controller.pid import PID


class LatencyPredictor:
    """Reconstructs the CURRENT position of the ball from a measurement that is N periods late.

    1. Model-based α-β observer: (x, v) is estimated at the time of the measurement, predicting
       with the model ẍ = c·g·θ and the command applied at that time, then correcting
       with the difference to the measurement: x += α·e, v += (β/T)·e.
    2. Projection: the model is integrated over the N periods of latency, with the last N
       commands, which the controller knows since it sent them itself.
    """

    def __init__(self, n_latency: int, period: float, model_gain: float, alpha: float = 0.5):
        self.n, self.T, self.g = n_latency, period, model_gain
        self.alpha = alpha
        self.beta = alpha ** 2 / (2 - alpha)          # standard tuning of an α-β filter
        self.commands = deque([0.0] * (n_latency + 1), maxlen=n_latency + 1)  # u_{k-N-1} ... u_{k-1}
        self.x = self.v = None

    def predict(self, measurement: float) -> float:
        if self.x is None:                            # first measurement: start from there, at rest
            self.x, self.v = measurement, 0.0
        else:
            # Observer prediction up to the time of the measurement (with u_{k-N-1})
            self.v += self.g * self.commands[0] * self.T
            self.x += self.v * self.T
            # Correction by the measurement
            e = measurement - self.x
            self.x += self.alpha * e
            self.v += self.beta / self.T * e
        # Projection up to now with u_{k-N} ... u_{k-1}
        x, v = self.x, self.v
        for u in list(self.commands)[1:]:
            v += self.g * u * self.T
            x += v * self.T
        return x

    def store(self, command: float) -> None:
        self.commands.append(command)


class PlateController:
    """Two independent PIDs (x and y axes), saturated at ±theta_max.

    Feedforward: for the ball to follow a trajectory with acceleration a_ref, the plate needs
    θ_ff = a_ref / (c·g) (linearized model). The PID only has to correct the deviations.
    model_gain = c·g = g / (1 + k), in (m/s²)/rad.

    latency_compensation = N > 0: the measurement is assumed to be N periods late, and the
    current position is reconstructed by LatencyPredictor before entering the PIDs.

    integration_zone: each PID integrates its error only while it is smaller than this distance (m).
    """

    def __init__(self, kp: float, ki: float, kd: float, theta_max: float,
                 tau_d: float, model_gain: float, feedforward: bool = True,
                 latency_compensation: int = 0, period: float = 0.02, integration_zone: float = math.inf):
        self.pid_x = PID(kp, ki, kd, -theta_max, theta_max, tau_d, integration_zone)
        self.pid_y = PID(kp, ki, kd, -theta_max, theta_max, tau_d, integration_zone)
        self.model_gain = model_gain
        self.feedforward = feedforward
        self.predictors = ([LatencyPredictor(latency_compensation, period, model_gain) for _ in range(2)]
                            if latency_compensation > 0 else None)

    def compute(self, setpoint, measurement, setpoint_acc, dt: float):
        """setpoint, measurement, setpoint_acc: (x, y) pairs. Returns (θx, θy) in rad."""
        if self.predictors is not None:
            measurement = (self.predictors[0].predict(measurement[0]), self.predictors[1].predict(measurement[1]))
        ffx = setpoint_acc[0] / self.model_gain if self.feedforward else 0.0
        ffy = setpoint_acc[1] / self.model_gain if self.feedforward else 0.0
        theta_x = self.pid_x.compute(setpoint[0], measurement[0], dt, ffx)
        theta_y = self.pid_y.compute(setpoint[1], measurement[1], dt, ffy)
        if self.predictors is not None:
            self.predictors[0].store(theta_x)
            self.predictors[1].store(theta_y)
        return theta_x, theta_y
