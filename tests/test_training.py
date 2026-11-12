"""Training environment of the AI brain: episodes, reward, interface, same robot as in the simulations."""
import math
import random
import time
from dataclasses import replace

import numpy as np
import pytest

from controller.policy import LAYOUT, MAX_LATENCY
from simulation.config import G, Circle, Config, Point, Step
from simulation.loop import policy_brain, simulate
from simulation.metrics import stoppable_speed
from training.environment import (EPISODE, BallPlateEnv, Reward, draw_episode, edge_distance, measure_speed,
                                  pd_policy)

SIMPLE = Config(physics="simple", mechanism=False)      # fast episodes, for what does not depend on the physics
CG = Config().plate.coefficient * G


def run(env, policy, periods: int):
    """Drives env with policy during `periods` control periods, through the resets.
    Returns the lists of observations, rewards and ends (terminated, truncated)."""
    observation, _ = env.reset()
    observations, rewards, ends = [observation], [], []
    for _ in range(periods):
        observation, reward, terminated, truncated, _ = env.step(policy(observation))
        rewards.append(reward)
        ends.append((terminated, truncated))
        if terminated or truncated:
            observation, _ = env.reset()
        observations.append(observation)
    return observations, rewards, ends


def do_nothing(observation):
    return 0.0, 0.0


# --- Episodes -------------------------------------------------------------------------------------
def test_edge_distance():
    assert edge_distance(-0.10, 0.0, 0.0, 0.15) == pytest.approx(0.25)          # the throw of the campaign
    assert edge_distance(0.0, 0.0, 1.0, 0.15) == pytest.approx(0.15)
    assert edge_distance(0.05, 0.05, math.pi / 2, 0.15) == pytest.approx(math.sqrt(0.15 ** 2 - 0.05 ** 2) - 0.05)


def test_episodes_are_drawn_within_their_ranges():
    rng, kinds = random.Random(0), set()
    for _ in range(400):
        cfg = draw_episode(rng)
        pm, tolerance = cfg.mujoco_params, 1 + 1e-9
        assert 2 / 5 <= pm.k_actual <= 2 / 3 and 0.0027 <= pm.ball_mass <= 0.027 * tolerance
        assert 0.25 <= pm.plate_mass <= 0.5 and 0.02 / tolerance <= pm.friction <= 1.0
        assert max(abs(cfg.plate.bias_x), abs(cfg.plate.bias_y)) <= math.radians(1)
        assert 1 <= cfg.latency <= MAX_LATENCY and 0.0001 <= cfg.sigma <= 0.001
        assert math.hypot(*cfg.start_position) <= 0.12 and cfg.duration == EPISODE
        # The setpoint stays at least 5 cm from the edge
        assert all(math.hypot(*cfg.trajectory(t)[:2]) <= 0.10 * tolerance for t in np.linspace(0, EPISODE, 21))
        # Thrown or pushed: up to the speed that the plate can just stop before the edge
        ball = "placed"
        if cfg.start_velocity != (0.0, 0.0):
            ball, origin, (vx, vy) = "thrown", cfg.start_position, cfg.start_velocity
        elif cfg.pushes:
            ((t, vx, vy),), ball = cfg.pushes, "pushed"
            origin = cfg.trajectory(t)[:2]
            assert 0.1 * EPISODE <= t <= 0.8 * EPISODE
        if ball != "placed":
            distance = edge_distance(*origin, math.atan2(vy, vx), 0.15)
            assert math.hypot(vx, vy) <= stoppable_speed(distance, math.radians(10), CG) * tolerance
        kinds.add((type(cfg.trajectory), ball))
    assert kinds == {(trajectory, ball) for trajectory in (Point, Step, Circle)
                     for ball in ("placed", "thrown", "pushed")}
    # Same state of the generator, same episode
    assert draw_episode(random.Random(5)) == draw_episode(random.Random(5)) != draw_episode(random.Random(6))


# --- Reward ---------------------------------------------------------------------------------------
def test_reward():
    reward, steady = Reward(), (0.2, 0.0)
    assert reward(0.0, steady, steady, False) == 1.0                    # on the setpoint, command unchanged
    assert reward(0.005, steady, steady, False) == pytest.approx(0.5 * (1 - 0.05 ** 2) + 0.5 * 0.5)
    assert reward(0.1, steady, steady, False) == pytest.approx(0.5 / 21)        # 10 cm away: the period is lost
    # The last millimeters count: from 15 mm to 2 mm, a quarter of a period (1 % with the parabola alone)
    assert reward(0.002, steady, steady, False) - reward(0.015, steady, steady, False) == pytest.approx(0.243, abs=0.001)
    assert reward(0.0, (1.0, 0.0), (-1.0, 0.0), False) == pytest.approx(1.0 - 0.05 * 4)   # jerk of the command
    assert reward(math.nan, steady, steady, True) == -100.0             # fall


# --- Environment ----------------------------------------------------------------------------------
def test_environment_interface():
    env = BallPlateEnv(seed=0, base=SIMPLE)
    observation, info = env.reset()
    assert observation.shape == (env.observation_size,) and observation.dtype == np.float32 and info == {}
    # The first observation comes with the first measurement, one latency after the start
    assert observation[LAYOUT["received"]] == pytest.approx(1 / 3)
    assert observation[LAYOUT["latency"]] == pytest.approx(env.sim.cfg.latency / MAX_LATENCY)
    assert env.sim.t == pytest.approx(env.sim.cfg.latency * 0.02)
    observation, reward, terminated, truncated, info = env.step((0.3, -2.0))
    assert observation[LAYOUT["commands"]][:2] == pytest.approx([0.3, -1.0])     # saturated at the maximum tilt
    assert env.sim.r.theta_x[-1] == pytest.approx(0.3 * math.radians(10))
    assert reward <= 1.0 and not terminated and not truncated and info["distance"] >= 0
    # An episode lasts 10 s: 500 control periods
    _, _, ends = run(BallPlateEnv(seed=0, base=SIMPLE), pd_policy, 500)
    assert ends.index((False, True)) == 499 and (True, False) not in ends


def test_same_seed_same_episodes():
    a, b, c = (run(BallPlateEnv(seed=seed, base=SIMPLE), pd_policy, 600) for seed in (3, 3, 4))
    assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0])) and a[1:] == b[1:]
    assert a[1] != c[1]
    # reset(seed) starts the sequence of episodes again
    env = BallPlateEnv(seed=9, base=SIMPLE)
    assert np.array_equal(env.reset(seed=3)[0], a[0][0])


def test_fall_ends_the_episode():
    # Without command, the calibration error of the plate rolls the ball off
    env = BallPlateEnv(seed=1, base=SIMPLE)
    observation, _ = env.reset()
    for _ in range(400):
        previous = observation
        observation, reward, terminated, truncated, info = env.step(do_nothing(observation))
        if terminated:
            break
        assert -100.0 < reward <= 1.0
    assert terminated and not truncated and reward == -100.0 and env.sim.r.fallen and math.isnan(info["distance"])
    assert np.array_equal(observation, previous)                # no measurement after the fall: the last one
    observation, _ = env.reset()                                # a new episode follows
    assert observation[LAYOUT["received"]] == pytest.approx(1 / 3) and not env.sim.r.fallen


def test_hand_written_policy_keeps_the_ball_of_the_mujoco_robot():
    # MuJoCo robot, 3 episodes: the PD holds the ball near its setpoint (a few mm away: it has no integral);
    # without command, it falls
    _, rewards, ends = run(BallPlateEnv(seed=3), pd_policy, 1500)
    assert (True, False) not in ends and sum(rewards) / len(rewards) > 0.5
    _, rewards, ends = run(BallPlateEnv(seed=3), do_nothing, 300)
    assert (True, False) in ends


def test_training_and_use_see_the_same_robot():
    # The same policy drives the robot from outside, through the environment (training), or as the brain of a
    # simulation (use): the same attempt, bit for bit
    env = BallPlateEnv(seed=4)
    observation, _ = env.reset()
    for _ in range(150):
        observation, _, terminated, _, _ = env.step(pd_policy(observation))
        assert not terminated
    trained = env.sim.r
    used = simulate(replace(env.sim.cfg, duration=3.5), brain=policy_brain(pd_policy))
    assert len(trained.theta_x) >= 150 and used.x[:len(trained.x)] == trained.x
    assert used.theta_x[:len(trained.theta_x)] == trained.theta_x
    assert used.actual_tilt[:len(trained.actual_tilt)] == trained.actual_tilt
    assert all(math.isnan(g) for g in used.model_gain)                # no model of the ball in this brain


def test_speed_measurement():
    assert measure_speed((0, time.time(), 0.3)) > 50          # control periods per second, one MuJoCo robot
