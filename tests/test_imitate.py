"""Imitation of the PID brain by the policy of the AI brain. Skipped without PyTorch (requirements-train.txt)."""
import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from controller.network import Network                              # noqa: E402 (after the skip)
from controller.policy import OBSERVATION_SIZE                       # noqa: E402
from simulation.config import Config                                 # noqa: E402
from simulation.loop import Simulation, policy_brain, simulate       # noqa: E402
from training import imitate as script                               # noqa: E402
from training.environment import draw_episode                        # noqa: E402
from training.evaluation import evaluate                             # noqa: E402

SIMPLE = Config(physics="simple", mechanism=False)


def lesson(policy=None, deviation=0.0, seconds=2.0):
    """Simulation of the first seconds of an episode, driven by a Lesson."""
    cfg = draw_episode(random.Random(3), SIMPLE)
    sim = Simulation(cfg, lambda cfg: script.Lesson(cfg, policy, deviation, random.Random(0)))
    for _ in range(round(seconds / cfg.dt_physics)):
        assert sim.step()
    return sim


def test_lesson_with_the_teacher_driving():
    sim = lesson()
    brain, r = sim.brain, sim.r
    # One observation of the pupil and one command of the teacher per decision, in units of the maximum tilt
    assert len(brain.observations) == len(brain.targets) == len(r.theta_x) - sim.cfg.latency
    assert brain.observations[0].shape == (OBSERVATION_SIZE,)
    assert brain.targets[-1] == pytest.approx((r.theta_x[-1] / sim.cfg.theta_max, r.theta_y[-1] / sim.cfg.theta_max))
    # The robot is driven as by the PID brain alone (to the rounding of the commands, written in units of the
    # maximum tilt in between)
    alone = simulate(sim.cfg)
    assert r.x == pytest.approx(alone.x[:len(r.x)], abs=1e-6)
    # With a deviation, the command sent is no longer the advice, and the teacher knows it
    sim = lesson(deviation=0.1)
    sent = sim.r.theta_x[-1]
    assert abs(sent / sim.cfg.theta_max - sim.brain.targets[-1][0]) > 1e-3
    assert sim.brain.teacher.predictors[0].commands[-1] == sent


def test_lesson_with_the_pupil_driving():
    sim = lesson(policy=lambda observation: (0.2, -0.1), seconds=0.5)
    brain, theta_max = sim.brain, sim.cfg.theta_max
    assert sim.r.theta_x[-1] == pytest.approx(0.2 * theta_max) and sim.r.theta_y[-1] == pytest.approx(-0.1 * theta_max)
    assert any(abs(target[0] - 0.2) > 1e-3 for target in brain.targets)             # the advice of the teacher
    assert set(brain.teacher.predictors[1].commands) <= {0.0, -0.1 * theta_max}     # told the commands really sent


def test_lessons_in_one_or_several_processes():
    seeds = [1, 2, 3, 4]
    together = script.lessons(seeds, None, 0.0, 1, SIMPLE)
    spread = script.lessons(seeds, None, 0.0, 2, SIMPLE)
    assert together[0].shape == (len(together[1]), OBSERVATION_SIZE) and together[1].shape[1] == 2
    assert together[0].dtype == together[1].dtype == np.float32 and 1500 < len(together[0]) <= 2000
    assert np.array_equal(together[0], spread[0]) and np.array_equal(together[1], spread[1])
    assert together[2:] == pytest.approx(spread[2:]) and 0.0 <= together[2] <= 1.0 and 0.0 < together[3] < 0.1
    # The pupil drives: other lessons
    layers = [(np.zeros((2, OBSERVATION_SIZE)), np.zeros(2))]
    assert not np.array_equal(script.lessons(seeds, layers, 0.0, 1, SIMPLE)[1][:1500], together[1][:1500])


def test_pupil_learns_and_gives_its_network(monkeypatch):
    monkeypatch.setattr(script, "BATCH", 250)
    rng = np.random.default_rng(0)
    observations = rng.normal(1.0, 2.0, (4000, OBSERVATION_SIZE)).astype(np.float32)
    weights = rng.normal(0.0, 0.05, (OBSERVATION_SIZE, 2)).astype(np.float32)
    targets = np.tanh(observations @ weights)
    pupil = script.Pupil(observations)
    errors = [pupil.fit(observations, targets, epochs=20) for _ in range(3)]
    assert errors[2] < errors[0] and errors[2] < 0.2 * float(targets.std())
    # The network given (NumPy) is the function learned (PyTorch), scaling of the inputs included
    with torch.no_grad():
        learned = pupil.net((torch.as_tensor(observations[:5]) - pupil.mean) / pupil.scale).numpy()
    network = Network(pupil.layers())
    assert np.array([network(o) for o in observations[:5]]) == pytest.approx(learned, abs=1e-5)


def test_imitation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "FOLDER", tmp_path)
    monkeypatch.setattr(script, "VALIDATION", 20)
    network = script.imitate("pupil", rounds=2, episodes=100, epochs=20, processes=1, base=SIMPLE)
    out = capsys.readouterr().out
    assert "teacher alone" in out and "round 1" in out and out.count("pupil   alone") == 2
    # The policy saved is the one returned; alone on the MuJoCo robot, it keeps the ball of the reference scenario
    observation = np.linspace(-1, 1, OBSERVATION_SIZE)
    assert np.array_equal(Network.load(tmp_path / "pupil" / "policy.npz")(observation), network(observation))
    marks = evaluate(policy_brain(network))
    assert not any(math.isnan(mark) for mark in marks.values()) and marks["circle_60ms_mm"] < 15.0
