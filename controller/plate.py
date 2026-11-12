"""2D plate controller: one PID per axis + feedforward
+ compensation of the measurement latency by prediction (optional, disabled by default)
+ conditional integration (optional, disabled by default)
+ online estimation of the model gain c·g (optional, disabled by default).

Independent of the simulator: it receives the setpoint, the measurement and the desired acceleration,
and returns the two tilts (θx, θy) to apply to the plate.
"""
from collections import deque
import math

from controller.adaptation import ModelGainEstimator
from controller.flight import FlightEstimator
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

    flight: estimator of the ball in the air (controller/flight.py). While the measured height is above its
    airborne_height, the PIDs only follow the ball and the plate is tilted for the bounce (aimed_tilt).

    estimator: estimator of the model gain (controller/adaptation.py). Given, it replaces model_gain, for the
    feedforward and the latency predictor, by its estimate of each period (the gains of the PID stay those of
    the nominal model).
    """

    def __init__(self, kp: float, ki: float, kd: float, theta_max: float,
                 tau_d: float, model_gain: float, feedforward: bool = True,
                 latency_compensation: int = 0, period: float = 0.02, integration_zone: float = math.inf,
                 flight: FlightEstimator = None, restitution: float = 0.87, bounce_gain: float = 0.5,
                 wait_for_speed: bool = False, estimator: ModelGainEstimator = None):
        self.pid_x = PID(kp, ki, kd, -theta_max, theta_max, tau_d, integration_zone, wait_for_speed)
        self.pid_y = PID(kp, ki, kd, -theta_max, theta_max, tau_d, integration_zone, wait_for_speed)
        self.model_gain = model_gain
        self.feedforward = feedforward
        self.predictors = ([LatencyPredictor(latency_compensation, period, model_gain) for _ in range(2)]
                            if latency_compensation > 0 else None)
        self.flight = flight
        self.restitution, self.bounce_gain = restitution, bounce_gain
        self.estimator = estimator

    def compute(self, setpoint, measurement, setpoint_acc, dt: float):
        """setpoint, setpoint_acc: (x, y) pairs; measurement: (x, y) or (x, y, height of the ball).
        Returns (θx, θy) in rad."""
        position = (measurement[0], measurement[1])
        airborne = False
        if self.flight is not None:
            self.flight.observe(tuple(measurement) if len(measurement) == 3 else (*measurement, 0.0))
            airborne = len(measurement) == 3 and measurement[2] > self.flight.airborne_height
        if self.estimator is not None:
            self.estimator.update(position, setpoint, airborne)
            self.model_gain = self.estimator.gain
            for predictor in self.predictors or ():
                predictor.g = self.model_gain
        if self.predictors is not None:
            measurement = (self.predictors[0].predict(measurement[0]), self.predictors[1].predict(measurement[1]))
        ffx = setpoint_acc[0] / self.model_gain if self.feedforward else 0.0
        ffy = setpoint_acc[1] / self.model_gain if self.feedforward else 0.0
        theta_x = self.pid_x.compute(setpoint[0], measurement[0], dt, ffx, hold=airborne)
        theta_y = self.pid_y.compute(setpoint[1], measurement[1], dt, ffy, hold=airborne)
        # In the air, θx and θy are the integral terms alone (they compensate the calibration error of the plate):
        # the plate stays level until the impact is estimated (2 measurements of the flight), then it is aimed
        if airborne and self.flight.impact() is not None:
            aim_x, aim_y = self.aimed_tilt(self.flight.impact(), setpoint)
            theta_x, theta_y = self.pid_x._saturate(theta_x + aim_x), self.pid_y._saturate(theta_y + aim_y)
        if self.predictors is not None:
            self.predictors[0].store(theta_x)
            self.predictors[1].store(theta_y)
        if self.flight is not None:
            self.flight.command((theta_x, theta_y))
        if self.estimator is not None:
            self.estimator.command((theta_x, theta_y))
        return theta_x, theta_y

    def applied(self, theta) -> None:
        """Another command than the one just computed was sent to the plate, (θx, θy): the memories of the
        commands sent are corrected (imitation, training/imitate.py: a pupil drives, this brain only advises)."""
        for predictor, command in zip(self.predictors or (), theta):
            predictor.commands[-1] = command
        for memory in (self.flight, self.estimator):
            if memory is not None:
                memory.commands[-1] = tuple(theta)

    def aimed_tilt(self, impact, setpoint):
        """Tilt of the plate at the impact so that the bounce sends the ball back toward the setpoint.

        A plate tilted by θ (small) changes the horizontal velocity of a ball hitting it at the vertical
        speed v_i by Δv = (1 + e)·v_i·θ (the normal component is reversed and multiplied by e). After the
        bounce, the next impact comes 2·e·v_i/g later: to cover the share bounce_gain of the distance d to the
        setpoint during that flight, the ball needs v = bounce_gain·d / (2·e·v_i/g), hence θ = (v - v_h) / ((1 + e)·v_i)."""
        _, p, v = impact
        v_i = -v[2]
        if v_i <= 0:
            return 0.0, 0.0
        flight_time = 2 * self.restitution * v_i / self.flight.g
        return tuple((self.bounce_gain * (setpoint[k] - p[k]) / flight_time - v[k]) / ((1 + self.restitution) * v_i)
                     for k in range(2))
