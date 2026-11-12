"""Training script of the AI brain. Skipped without the training libraries (requirements-train.txt)."""
import csv
import math

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from controller.policy import OBSERVATION_SIZE          # noqa: E402 (after the skip)
from training import train as script                    # noqa: E402
from training.environment import pd_policy              # noqa: E402


def rows(folder):
    with open(folder / "progress.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def short(tmp_path, monkeypatch):
    """Trainings of a few hundred periods, written in a temporary folder."""
    monkeypatch.setattr(script, "FOLDER", tmp_path)
    monkeypatch.setattr(script, "SETTINGS", {**script.SETTINGS, "n_steps": 64, "batch_size": 64, "n_epochs": 1})
    return tmp_path


def test_gym_environment():
    env = script.GymEnv(seed=0)
    observation, info = env.reset()
    assert env.observation_space.shape == (OBSERVATION_SIZE,) and env.observation_space.contains(observation)
    assert env.action_space.shape == (2,) and info == {}
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(observation) and reward <= 1.0 and info["fallen"] == terminated


def test_evaluation_on_the_reference_scenario():
    # The hand-written PD keeps the ball, without the precision of the PID brain (no integral, no feedforward)
    marks = script.evaluate(pd_policy)
    assert list(marks) == ["return_20ms_mm", "circle_20ms_mm", "return_60ms_mm", "circle_60ms_mm"]
    assert all(1.0 < mark < 50.0 for mark in marks.values())
    # Without command, the ball falls: no mark
    assert all(math.isnan(mark) for mark in script.evaluate(lambda observation: (0.0, 0.0)).values())


def test_training_saves_and_resumes(short, capsys):
    script.train("trial", periods=128, processes=1, every=64)
    folder = short / "trial"
    assert (folder / "model.zip").exists() and (folder / "rewards.pkl").exists()
    assert [int(row["periods"]) for row in rows(folder)] == [64, 128]
    assert "Training" in capsys.readouterr().out
    # The same command with more periods goes on from the saved training
    script.train("trial", periods=256, processes=1, every=64)
    progress = rows(folder)
    assert [int(row["periods"]) for row in progress] == [64, 128, 192, 256]
    assert float(progress[-1]["hours"]) >= float(progress[1]["hours"]) >= 0.0     # the hours add up
    assert all(0.0 < float(row["deviation"]) < 1.0 for row in progress)
    assert "Resuming" in capsys.readouterr().out
    # Nothing left to do
    script.train("trial", periods=256, processes=1, every=64)
    assert "Already trained" in capsys.readouterr().out and len(rows(folder)) == 4


def test_same_seed_same_training(short):
    for name in ("a", "b"):
        script.train(name, periods=128, processes=1, every=128)
    a, b = (script.PPO.load(short / name / "model.zip", device="cpu") for name in ("a", "b"))
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    assert np.array_equal(script.policy_of(a)(observation), script.policy_of(b)(observation))


def test_training_on_several_processes(short):
    script.train("parallel", periods=128, processes=2, every=128)        # 2 robots, 64 periods each
    assert [int(row["periods"]) for row in rows(short / "parallel")] == [128]
