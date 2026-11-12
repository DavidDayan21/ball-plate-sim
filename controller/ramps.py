"""Rule of the fast ramps of the motors: the same for every brain."""
import math


class FastRamps:
    """The motors may use their fast ramp (simulation/motor.py) only while the ball moves faster than fast_speed.

    A slow ball has no need of the fast ramp, which would throw it off the plate: the plate under it would
    accelerate downward faster than g (r·θ̈ > g, beyond about 100 rad/s² of plate 10 cm from the center).

    The speed is taken from the last two measured positions, as they arrive: without the lag of a filter (a pushed
    ball gets the fast ramp at once) and without the amplification of the noise by a prediction.
    """

    def __init__(self, fast_speed: float):
        self.fast_speed = fast_speed           # (m/s); 0: whatever the speed
        self.allowed = False                   # no speed estimated yet
        self.previous = None                   # last measured (x, y)

    def update(self, position, dt: float) -> bool:
        """position: (x, y) just measured, dt after the previous one. Returns whether the fast ramps are allowed."""
        speed = (0.0 if self.previous is None else
                 math.hypot(position[0] - self.previous[0], position[1] - self.previous[1]) / dt)
        self.previous = (position[0], position[1])
        self.allowed = speed >= self.fast_speed
        return self.allowed
