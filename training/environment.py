"""Training environment of the AI brain: episodes of the simulated robot, drawn at random, every period rewarded.

The policy of the AI brain (controller/policy.py) is learned by reinforcement: it tries, receives a reward at every
control period, and improves over millions of periods. This module provides them. BallPlateEnv has the interface
of the Gymnasium environments (reset, step) without depending on that library: the training script wraps it.

An episode (draw_episode) lasts 10 s. Drawn for each one:
    - the robot and the ball, which the brain does not know, between the variants of the robustness section of
      the campaign: inertia of the ball from solid to hollow, mass up to 10 times a ping-pong ball, friction from
      0.02 to 1, mass of the plate up to twice, calibration error up to 1° on each axis;
    - the sensor: latency of 1 to 5 periods (20 to 100 ms, known to the brain), noise of 0.1 to 1 mm (the camera
      has about 0.2 mm, but rendering its images for millions of periods would take too long);
    - the scenario: ball placed, thrown or pushed (up to the physical limit, the speed that the plate can just
      stop before the edge); setpoint fixed (at the center half of the time), jumping to another point, or
      following a circle.
A ball dropped from above is left for a later version.

Reward of a control period (Reward): 1 - (distance to the setpoint / 10 cm)², minus the jerk of the command; a
fall ends the episode and costs a penalty. The distance is the true one, at the end of the period (the reward is
not seen by the brain). The constant 1 only matters at a fall: it then loses all the periods to come.

    python -m training.environment     speed of the environment on this machine
"""
from dataclasses import dataclass, replace
import math
import os
import random
import time

from controller.policy import LAYOUT, LENGTH, MAX_LATENCY, OBSERVATION_SIZE, SPEED
from simulation.config import G, Circle, Config, Point, Step
from simulation.loop import Simulation, policy_brain
from simulation.metrics import stoppable_speed
from simulation.parallel import execute

EPISODE = 10.0                    # (s)
START_RADIUS = 0.12               # (m) the ball starts within this distance of the center
SETPOINT_RADIUS = 0.08            # (m) fixed setpoints: within this distance of the center
CIRCLE_RADIUS = (0.03, 0.08)      # (m)
CIRCLE_PERIOD = (3.0, 6.0)        # (s) the reference circle: 6 cm in 4 s
CIRCLE_CENTER = 0.02              # (m) center of the circles: within this distance of the center of the plate


def edge_distance(x: float, y: float, direction: float, radius: float) -> float:
    """Distance from (x, y) to the edge of a plate of this radius, along the angle `direction`."""
    ux, uy = math.cos(direction), math.sin(direction)
    along = x * ux + y * uy
    return -along + math.sqrt(along ** 2 + radius ** 2 - x * x - y * y)


def draw_episode(rng: random.Random, base: Config = None, duration: float = EPISODE) -> Config:
    """Configuration of an episode drawn with rng, from `base` (the default Config: MuJoCo robot, synthetic
    sensor)."""
    base = base or Config()
    two_pi = 2 * math.pi

    def point(radius: float):
        """Uniform in the disk of this radius."""
        r, a = radius * math.sqrt(rng.random()), rng.uniform(0, two_pi)
        return r * math.cos(a), r * math.sin(a)

    def target():
        return (0.0, 0.0) if rng.random() < 0.5 else point(SETPOINT_RADIUS)

    def limit(position, direction: float) -> float:
        """Speed that the plate can just stop before the edge (nominal ball)."""
        distance = edge_distance(*position, direction, base.plate.plate_radius)
        return stoppable_speed(distance, base.theta_max, base.plate.coefficient * G)

    # The robot and the ball
    pm = base.mujoco_params
    pm = replace(pm, k_actual=rng.uniform(2 / 5, 2 / 3), ball_mass=pm.ball_mass * 10 ** rng.uniform(0, 1),
                 plate_mass=pm.plate_mass * rng.uniform(1, 2), friction=10 ** rng.uniform(math.log10(0.02), 0))
    plate = replace(base.plate, bias_x=math.radians(rng.uniform(-1, 1)), bias_y=math.radians(rng.uniform(-1, 1)))
    # The sensor
    latency, sigma, seed = rng.randint(1, MAX_LATENCY), rng.uniform(0.0001, 0.001), rng.getrandbits(32)
    # The setpoint
    kind = rng.choice(("point", "step", "circle"))
    if kind == "point":
        trajectory = Point(*target())
    elif kind == "step":
        trajectory = Step(target(), target(), rng.uniform(0.2, 0.8) * duration)
    else:
        cx, cy = point(CIRCLE_CENTER)
        trajectory = Circle(rng.uniform(*CIRCLE_RADIUS), rng.choice((-1, 1)) * rng.uniform(*CIRCLE_PERIOD),
                            rng.uniform(0, two_pi), cx, cy)
    # The ball
    kind = rng.choice(("placed", "thrown", "pushed"))
    position, velocity, pushes = point(START_RADIUS), (0.0, 0.0), ()
    direction, share = rng.uniform(0, two_pi), rng.random()
    if kind == "thrown":
        speed = share * limit(position, direction)
        velocity = (speed * math.cos(direction), speed * math.sin(direction))
    elif kind == "pushed":                          # where the ball should be by then: at its setpoint
        t = rng.uniform(0.1, 0.8) * duration
        speed = share * limit(trajectory(t)[:2], direction)
        pushes = ((t, speed * math.cos(direction), speed * math.sin(direction)),)
    return replace(base, mujoco_params=pm, plate=plate, latency=latency, sigma=sigma, seed=seed,
                   trajectory=trajectory, start_position=position, start_velocity=velocity, pushes=pushes,
                   duration=duration)


@dataclass(frozen=True)
class Reward:
    """Reward of a control period. action, previous_action: in units of theta_max (controller/policy.py)."""
    scale: float = 0.1            # (m) distance to the setpoint that costs a whole period
    jerk: float = 0.05            # weight of the squared change of the action between two periods
    fall: float = 100.0           # penalty of a fall: as much as 2 s on the setpoint bring

    def __call__(self, distance: float, action, previous_action, fallen: bool) -> float:
        if fallen:
            return -self.fall
        change = (action[0] - previous_action[0]) ** 2 + (action[1] - previous_action[1]) ** 2
        return 1.0 - (distance / self.scale) ** 2 - self.jerk * change


class BallPlateEnv:
    """One robot, one episode after the other. The policy is outside: it receives the observation of the brain
    (controller/policy.py) and answers with its action.

        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(action)

    terminated: the ball has fallen; truncated: the episode has reached its duration. Same seed, same episodes.
    """
    observation_size = OBSERVATION_SIZE
    action_size = 2

    def __init__(self, seed: int = 0, base: Config = None, duration: float = EPISODE, reward: Reward = Reward()):
        self.rng = random.Random(seed)
        self.base, self.duration, self.reward = base or Config(), duration, reward
        self.max_steps = round(duration / self.base.control_period)
        self.sim = None                       # the simulation of the current episode; its settings: self.sim.cfg

    def reset(self, seed: int = None):
        """Starts a new episode. Returns (observation, info) at the first measurement, one latency after the
        start (until then, the plate keeps the command 0, as with the PID brain)."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.observation = None
        while self.observation is None:       # a ball thrown too close to the edge falls before the first image
            self.sim = Simulation(draw_episode(self.rng, self.base, self.duration), policy_brain())
            self._observe()
        self.steps = 0
        self.previous_action = (0.0, 0.0)
        return self.observation, {}

    def step(self, action):
        """Applies the action during one control period."""
        sim, brain = self.sim, self.sim.brain
        sim.theta = brain.act(action)
        sim.actuate()
        fallen = not self._observe()
        self.steps += 1
        r = sim.r
        distance = math.nan if fallen else math.hypot(r.x[-1] - r.x_ref[-1], r.y[-1] - r.y_ref[-1])
        applied = brain.commands[-1]
        reward = self.reward(distance, applied, self.previous_action, fallen)
        self.previous_action = applied
        truncated = not fallen and self.steps >= self.max_steps
        return self.observation, reward, fallen, truncated, {"distance": distance}

    def _observe(self) -> bool:
        """Advances to the next decision of the brain and stores its observation. False if the ball has fallen
        (the last observation is kept)."""
        inputs = self.sim.next_decision()
        if inputs is None:
            return False
        self.observation = self.sim.brain.observe(*inputs, self.sim.cfg.control_period)
        return True


# --- A policy written by hand, to test the environment and to measure its speed ------------------
_KP, _, _KD = Config().pid_gains()
_THETA_MAX = Config().theta_max


def pd_policy(observation):
    """PD on the newest measurement, with the gains of the PID brain; no integral, no latency compensation."""
    error = observation[LAYOUT["setpoint"]] - observation[LAYOUT["position"]]
    speed = observation[LAYOUT["speeds"]][:2]
    return (_KP * LENGTH * error - _KD * SPEED * speed) / _THETA_MAX


def measure_speed(job) -> float:
    """Control periods simulated per second by one environment driven by pd_policy, resets included.
    job = (seed, start, duration): the measurement starts at the time `start` (time.time()) and lasts `duration`
    seconds, so that several processes measure at the same time."""
    seed, start, duration = job
    env = BallPlateEnv(seed)
    observation, _ = env.reset()              # also loads MuJoCo: before the measurement
    time.sleep(max(0.0, start - time.time()))
    periods, begin = 0, time.perf_counter()
    while time.perf_counter() - begin < duration:
        observation, _, terminated, truncated, _ = env.step(pd_policy(observation))
        periods += 1
        if terminated or truncated:
            observation, _ = env.reset()
    return periods / (time.perf_counter() - begin)


if __name__ == "__main__":
    DURATION, LEAD = 10.0, 15.0               # (s) measurement; time left to the processes to start
    cores = os.cpu_count() or 1
    print(f"Speed of the training environment on this machine ({cores} logical cores), in control periods per "
          "second (one period: 20 ms of the robot):")
    best = 0.0
    for processes in sorted({1, max(1, cores // 2), cores}):
        start = time.time() + (LEAD if processes > 1 else 0.0)
        rates = execute(measure_speed, [(seed, start, DURATION) for seed in range(processes)], processes)
        best = max(best, sum(rates))
        print(f"  {processes:3d} process(es): {sum(rates):7.0f} periods/s ({min(rates):.0f} to {max(rates):.0f} each)",
              flush=True)
    print(f"At the best rate: 10 million periods in {1e7 / best / 3600:.1f} h, 100 million in "
          f"{1e8 / best / 3600:.0f} h (simulation alone, without the training computations).")
