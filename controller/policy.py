"""Brain driven by a policy: a function from what the brain knows (the observation) to its command (the action).

The policy of the AI brain is a neural network, learned in simulation (training/). This class is the part of that
brain which is not learned: its memory and the layout of its observation, the same during the training
(training/environment.py) and on the robot.

It plays by the rules of the PID brain and knows what the PID knows: the measurements as they arrive, N periods
late (N is known), the commands it has sent, the setpoint and its acceleration, the maximum tilt. It knows neither
the ball (inertia, mass, friction) nor the calibration error of the plate.

Observation: OBSERVATION_SIZE numbers of comparable sizes (LAYOUT gives the place of each group).
    position               newest measured (x, y), in units of LENGTH
    speeds                 displacements between the last 3 measurements, newest first, as speeds (x, y) in units of
                           SPEED: the 3 measurements themselves, written so that the policy gets numbers of
                           comparable sizes (0 while a measurement is missing)
    heights                last 3 measured heights of the ball, newest first, in units of HEIGHT
    commands               last COMMANDS actions, newest first: a measurement N periods late does not show yet the
                           effect of the last N commands (the predictor of the PID uses N + 1 of them)
    setpoint               (x, y), in units of LENGTH
    setpoint_speed         its displacement since the previous period, as a speed in units of SPEED, bounded at
                           ±SETPOINT_SPEED_LIMIT (a jump of the setpoint is not a speed)
    setpoint_acceleration  in units of ACCELERATION
    latency                N / MAX_LATENCY
    received               measurements received so far, min(count, 3) / 3: no speed is known at the first one

Action: (ax, ay) in [-1, 1], the command (θx, θy) in units of theta_max; beyond, it is saturated.
"""
from collections import deque
import math

import numpy as np

MEASUREMENTS = 3
MAX_LATENCY = 5                  # (control periods) 100 ms
COMMANDS = MAX_LATENCY + 1
LENGTH = 0.1                     # (m)
SPEED = 0.2                      # (m/s)
HEIGHT = 0.02                    # (m) radius of the ball
ACCELERATION = 1.0               # (m/s²) about what the plate gives the ball at its maximum tilt (c·g·θmax)
SETPOINT_SPEED_LIMIT = 2.5       # (units of SPEED) 0.5 m/s


def _layout(**sizes):
    """Slice of each group in the observation, and total size."""
    slices, start = {}, 0
    for name, size in sizes.items():
        slices[name] = slice(start, start + size)
        start += size
    return slices, start


LAYOUT, OBSERVATION_SIZE = _layout(position=2, speeds=2 * (MEASUREMENTS - 1), heights=MEASUREMENTS,
                                   commands=2 * COMMANDS, setpoint=2, setpoint_speed=2, setpoint_acceleration=2,
                                   latency=1, received=1)


class PolicyController:
    """policy: function observation -> action. None: the brain is driven from outside (training), which calls
    observe() then act() instead of compute()."""
    model_gain = math.nan            # no model of the ball (simulation/loop.py)

    def __init__(self, policy, theta_max: float, latency: int):
        if not 0 <= latency <= MAX_LATENCY:
            raise ValueError(f"Latency of {latency} periods: the policy knows 0 to {MAX_LATENCY}")
        self.policy, self.theta_max, self.latency = policy, theta_max, latency
        self.measurements = deque(maxlen=MEASUREMENTS)                    # (x, y, height), oldest first
        self.commands = deque([(0.0, 0.0)] * COMMANDS, maxlen=COMMANDS)   # actions, oldest first
        self.received = 0
        self.previous_setpoint = None

    def observe(self, setpoint, measurement, setpoint_acc, dt: float) -> np.ndarray:
        """Stores the measurement, (x, y) or (x, y, height of the ball), and returns the observation."""
        self.measurements.append(tuple(measurement) if len(measurement) == 3 else (*measurement, 0.0))
        self.received += 1
        m = list(self.measurements)[::-1]                                 # newest first
        m += [m[-1]] * (MEASUREMENTS - len(m))                            # missing ones: no displacement
        previous = setpoint if self.previous_setpoint is None else self.previous_setpoint
        self.previous_setpoint = (setpoint[0], setpoint[1])
        observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
        observation[LAYOUT["position"]] = (m[0][0] / LENGTH, m[0][1] / LENGTH)
        observation[LAYOUT["speeds"]] = [(a[k] - b[k]) / dt / SPEED for a, b in zip(m, m[1:]) for k in range(2)]
        observation[LAYOUT["heights"]] = [p[2] / HEIGHT for p in m]
        observation[LAYOUT["commands"]] = [a for command in reversed(self.commands) for a in command]
        observation[LAYOUT["setpoint"]] = (setpoint[0] / LENGTH, setpoint[1] / LENGTH)
        observation[LAYOUT["setpoint_speed"]] = [
            max(-SETPOINT_SPEED_LIMIT, min(SETPOINT_SPEED_LIMIT, (setpoint[k] - previous[k]) / dt / SPEED))
            for k in range(2)]
        observation[LAYOUT["setpoint_acceleration"]] = (setpoint_acc[0] / ACCELERATION,
                                                        setpoint_acc[1] / ACCELERATION)
        observation[LAYOUT["latency"]] = self.latency / MAX_LATENCY
        observation[LAYOUT["received"]] = min(self.received, MEASUREMENTS) / MEASUREMENTS
        return observation

    def act(self, action):
        """Stores the action, saturated, and returns the command (θx, θy) in rad."""
        ax, ay = (float(a) for a in action)
        if math.isnan(ax + ay):
            raise ValueError(f"Action of the policy: {action}")
        ax, ay = max(-1.0, min(1.0, ax)), max(-1.0, min(1.0, ay))
        self.commands.append((ax, ay))
        return ax * self.theta_max, ay * self.theta_max

    def compute(self, setpoint, measurement, setpoint_acc, dt: float):
        """Same interface as the PID brain (controller/plate.py). Returns (θx, θy) in rad."""
        return self.act(self.policy(self.observe(setpoint, measurement, setpoint_acc, dt)))
