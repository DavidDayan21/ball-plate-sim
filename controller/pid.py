"""Discrete PID controller.

It knows nothing about the simulator: it receives a setpoint and a measurement, and returns a command.
The same code is meant to run in the simple physics, in MuJoCo, then on the real robot.
"""
import math


class PID:
    """u = Kp·e + Ki·∫e·dt + Kd·de/dt + u_ff,   with e = setpoint - measurement.

    - Derivative: finite difference on the error, (e_n - e_{n-1}) / dt, filtered by a
      first-order low-pass with time constant tau_d (tau_d = 0: no filter).
    - Saturation: the output is clamped to [output_min, output_max].
    - Anti-windup: when the output is saturated and the error pushes further
      into saturation, the integral is frozen.
    - Conditional integration: the error is integrated only while |e| < integration_zone. Far from
      the setpoint, the proportional and derivative terms bring the measurement back; an integral
      built up during that transient would only make it overshoot.
    - Feedforward: u_ff is added before saturation.
    - Start with the speed (wait_for_speed): at the first measurement the speed of the measured object is
      unknown, and acting on its position alone would push a moving ball the wrong way. The output is then
      the feedforward and the integral term only; at the second measurement, the filtered derivative starts
      at the raw speed of the first two measurements instead of 0 (no lag of the filter at the start).

    With the default values (no filter, no saturation, integration everywhere, u_ff = 0), this is exactly
    the plain textbook PID: proportional, rectangle-rule integral and unfiltered finite-difference derivative.

    Units for one axis of the plate (a ball on a beam): e in m, u = θ in rad, hence Kp in rad/m.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 output_min: float = -math.inf, output_max: float = math.inf,
                 tau_d: float = 0.0, integration_zone: float = math.inf, wait_for_speed: bool = False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.tau_d = tau_d
        self.integration_zone = integration_zone
        self.wait_for_speed = wait_for_speed
        self.reset()

    def reset(self) -> None:
        """Resets the controller memory."""
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0
        self.speed_unknown = False       # wait_for_speed: between the first and the second measurement
        self.terms = (0.0, 0.0, 0.0)     # (P term, I term, D term) of the last computation

    def _saturate(self, u: float) -> float:
        return min(max(u, self.output_min), self.output_max)

    def compute(self, setpoint: float, measurement: float, dt: float, feedforward: float = 0.0,
                hold: bool = False) -> float:
        """Computes the command for one control step of duration dt.
        hold: the PID only follows the measurement (derivative), integrates nothing and returns its
        integral term alone (ball in the air: the plate is driven by the aimed bounce)."""
        e = setpoint - measurement
        if self.wait_for_speed and self.previous_error is None:
            self.previous_error, self.speed_unknown = e, True
            self.terms = (0.0, self.ki * self.integral, 0.0)
            return self._saturate(self.ki * self.integral + feedforward)

        # Filtered derivative: d <- a·d + (1 - a)·d_raw, with a = tau_d / (tau_d + dt)
        raw = 0.0 if self.previous_error is None else (e - self.previous_error) / dt
        if self.speed_unknown:                      # second measurement: the filter starts at the raw speed
            self.derivative, self.speed_unknown = raw, False
        a = self.tau_d / (self.tau_d + dt)
        self.derivative = a * self.derivative + (1 - a) * raw
        self.previous_error = e
        if hold:
            self.terms = (0.0, self.ki * self.integral, 0.0)
            return self._saturate(self.ki * self.integral)

        up = self.kp * e
        ud = self.kd * self.derivative
        candidate_integral = self.integral + e * dt if abs(e) < self.integration_zone else self.integral
        u = up + self.ki * candidate_integral + ud + feedforward
        u_sat = self._saturate(u)

        # Anti-windup: saturated AND the error pushes in the direction of saturation -> freeze the integral
        if u_sat != u and (u - u_sat) * e > 0:
            u_sat = self._saturate(up + self.ki * self.integral + ud + feedforward)
        else:
            self.integral = candidate_integral

        self.terms = (up, self.ki * self.integral, ud)
        return u_sat
