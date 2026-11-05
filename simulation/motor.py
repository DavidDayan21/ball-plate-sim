"""Command model of a position-controlled stepper motor.

Assumptions: the requested position is rounded to the nearest step (or microstep) and clamped
by limits; the motor reaches it with a bounded speed and, if max_acceleration is finite,
a bounded acceleration (trapezoidal profile, like stepper motor drivers).

    max_acceleration = inf: "square-wave" profile (max speed immediately)
    finite max_acceleration: acceleration and braking ramps

Torque and inertia are not modeled here: in MuJoCo, this profile is the setpoint
of the motor servo, which has a limited torque.
"""
import math


class Motor:
    def __init__(self, initial_angle: float, max_speed: float, resolution: float,
                 angle_min: float = math.radians(-60), angle_max: float = math.radians(60),
                 max_acceleration: float = math.inf):
        self.position = initial_angle      # angle (rad)
        self.velocity = 0.0                # velocity (rad/s), used with the ramps
        self.target = initial_angle        # requested angle, rounded to the step (rad)
        self.max_speed = max_speed         # rad/s
        self.max_acceleration = max_acceleration  # rad/s²
        self.resolution = resolution       # rad per step
        self.angle_min, self.angle_max = angle_min, angle_max

    def request(self, angle: float) -> None:
        """New angle setpoint: rounded to the nearest step, then clamped by the limits."""
        quantized = round(angle / self.resolution) * self.resolution
        self.target = min(max(quantized, self.angle_min), self.angle_max)

    def advance(self, dt: float) -> float:
        """Moves the motor toward its target for dt. Returns the angle."""
        gap = self.target - self.position
        if math.isinf(self.max_acceleration):
            max_step = self.max_speed * dt
            self.position += max(-max_step, min(max_step, gap))
            return self.position
        # Ramp: aim for the highest speed that still allows stopping on the target
        # when braking at a_max, i.e. v = √(2·a_max·|gap|), without exceeding max_speed;
        # and the velocity changes by at most a_max·dt per step.
        v_target = math.copysign(min(self.max_speed, math.sqrt(2 * self.max_acceleration * abs(gap))), gap)
        dv = self.max_acceleration * dt
        self.velocity += max(-dv, min(dv, v_target - self.velocity))
        self.position += self.velocity * dt
        return self.position
