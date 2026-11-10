"""Command model of a position-controlled stepper motor.

Assumptions: the requested position is rounded to the nearest step (or microstep; resolution 0: not
rounded) and clamped by limits; the motor reaches it with a bounded speed and, if max_acceleration is finite,
a bounded acceleration (trapezoidal profile, like stepper motor drivers).

    max_acceleration = inf: "square-wave" profile (max speed immediately)
    finite max_acceleration: acceleration and braking ramps
    fast_move finite: two ramps. A large move (gap to the target above fast_move) uses fast_acceleration
    until the motor has stopped on its target; the small corrections keep the gentle max_acceleration,
    so that the plate does not follow the noise of the command (which would make the ball hop).

Torque and inertia are not modeled here: in MuJoCo, this profile is the setpoint
of the motor servo, which has a limited torque.

The angle-driven plate (no arms) uses the same profile on its two angles, without steps: a plate that
jumped to every command would make the ball bounce.
"""
import math


def motors(cfg, initial_angles, resolution: float) -> list:
    """One Motor per initial angle, with the speed and the ramps of the configuration."""
    return [Motor(a, cfg.max_speed, resolution, max_acceleration=cfg.max_acceleration,
                  fast_acceleration=cfg.fast_acceleration, fast_move=cfg.fast_move) for a in initial_angles]


class Motor:
    def __init__(self, initial_angle: float, max_speed: float, resolution: float,
                 angle_min: float = math.radians(-60), angle_max: float = math.radians(60),
                 max_acceleration: float = math.inf, fast_acceleration: float = math.inf,
                 fast_move: float = math.inf):
        self.position = initial_angle      # angle (rad)
        self.velocity = 0.0                # velocity (rad/s), used with the ramps
        self.target = initial_angle        # requested angle, rounded to the step (rad)
        self.max_speed = max_speed         # rad/s
        self.max_acceleration = max_acceleration  # rad/s²
        self.fast_acceleration, self.fast_move = fast_acceleration, fast_move
        self.fast = False                  # a large move is in progress
        self.resolution = resolution       # rad per step
        self.angle_min, self.angle_max = angle_min, angle_max

    def request(self, angle: float) -> None:
        """New angle setpoint: rounded to the nearest step, then clamped by the limits."""
        quantized = round(angle / self.resolution) * self.resolution if self.resolution else angle
        self.target = min(max(quantized, self.angle_min), self.angle_max)

    def advance(self, dt: float) -> float:
        """Moves the motor toward its target for dt. Returns the angle."""
        gap = self.target - self.position
        if math.isinf(self.max_acceleration):
            max_step = self.max_speed * dt
            self.position += max(-max_step, min(max_step, gap))
            return self.position
        if abs(gap) > self.fast_move:
            self.fast = True
        elif self.fast and abs(gap) < self.fast_move / 4 and abs(self.velocity) < 0.5:
            self.fast = False              # stopped near the target: the large move is over
        a_max = self.fast_acceleration if self.fast else self.max_acceleration
        # Ramp: aim for the highest speed that still allows stopping on the target
        # when braking at a_max, i.e. v = √(2·a_max·|gap|), without exceeding max_speed;
        # and the velocity changes by at most a_max·dt per step.
        v_target = math.copysign(min(self.max_speed, math.sqrt(2 * a_max * abs(gap))), gap)
        dv = a_max * dt
        self.velocity += max(-dv, min(dv, v_target - self.velocity))
        self.position += self.velocity * dt
        return self.position
