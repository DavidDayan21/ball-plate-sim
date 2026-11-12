"""Training script of the AI brain. Skipped without the training libraries (requirements-train.txt)."""
import csv
import math

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from controller.policy import LAYOUT, OBSERVATION_SIZE  # noqa: E402 (after the skip)
from training import train as script                    # noqa: E402
from training.environment import pd_policy              # noqa: E402


def rows(folder):
    with open(folder / "progress.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def short(tmp_path, monkeypatch):
    """Trainings of a few hundred periods, written in a temporary folder."""
    monkeypatch.setattr(script, "FOLDER", tmp_path)
    monkeypatch.setattr(script, "SETTINGS", {**script.SETTINGS, "batch_size": 64, "n_epochs": 1})
    monkeypatch.setattr(script, "ROLLOUT", 64)
    return tmp_path


def play(robots, steps: int):
    """Observations, rewards and ends of `steps` periods with a fixed sequence of actions."""
    rng = np.random.default_rng(0)
    out = [robots.reset()]
    for _ in range(steps):
        observations, rewards, dones, infos = robots.step(rng.uniform(-1, 1, (robots.num_envs, 2)).astype(np.float32))
        out += [observations, rewards, dones]
    robots.close()
    return out, infos


def test_robots():
    robots = script.Robots(processes=1, per_process=3)
    assert robots.num_envs == 3 and robots.observation_space.shape == (OBSERVATION_SIZE,)
    assert robots.action_space.shape == (2,)
    observations = robots.reset()
    assert observations.shape == (3, OBSERVATION_SIZE) and observations.dtype == np.float32
    assert not np.array_equal(observations[0], observations[1])           # each robot has its own episodes
    observations, rewards, dones, infos = robots.step(np.zeros((3, 2), dtype=np.float32))
    assert observations.shape == (3, OBSERVATION_SIZE) and rewards.shape == dones.shape == (3,)
    assert all(info["fallen"] == done for info, done in zip(infos, dones))
    assert 0.0 < robots.busy[0] <= robots.waiting                       # seconds computing, seconds waited
    # Without command, the balls fall: the next episode starts at once, the last observation is in the info
    for _ in range(400):
        previous = observations
        observations, rewards, dones, infos = robots.step(np.zeros((3, 2), dtype=np.float32))
        if dones.any():
            break
    k = int(np.argmax(dones))
    assert infos[k]["fallen"] and not infos[k]["TimeLimit.truncated"] and rewards[k] == -100.0
    assert np.array_equal(infos[k]["terminal_observation"], previous[k])
    assert observations[k][LAYOUT["received"]] == pytest.approx(1 / 3)
    robots.close()


def test_robots_are_the_same_in_one_or_several_processes():
    together, _ = play(script.Robots(processes=1, per_process=4), 40)
    spread, infos = play(script.Robots(processes=2, per_process=2), 40)
    assert all(np.array_equal(a, b) for a, b in zip(together, spread)) and len(infos) == 4


def test_evaluation_on_the_reference_scenario():
    # The hand-written PD keeps the ball, without the precision of the PID brain (no integral, no feedforward)
    marks = script.evaluate(pd_policy)
    assert list(marks) == ["return_20ms_mm", "circle_20ms_mm", "return_60ms_mm", "circle_60ms_mm"]
    assert all(1.0 < mark < 50.0 for mark in marks.values())
    # Without command, the ball falls: no mark
    assert all(math.isnan(mark) for mark in script.evaluate(lambda observation: (0.0, 0.0)).values())


def test_training_saves_and_resumes(short, capsys):
    script.train("trial", periods=128, processes=1, every=64, per_process=1)
    folder = short / "trial"
    assert (folder / "model.zip").exists() and (folder / "rewards.pkl").exists()
    assert [int(row["periods"]) for row in rows(folder)] == [64, 128]
    out = capsys.readouterr().out
    assert "Training" in out
    # Where the time went: 5 shares of 0 to 100 %, one line per evaluation
    lines = [line for line in out.splitlines() if "time:" in line]
    assert len(lines) == 2 and all(name in lines[0] for name in ("simulation", "exchanges", "policy", "learning"))
    shares = [float(part.split()[-2]) for part in lines[1].split("time:")[1].split(",")]
    assert len(shares) == 5 and all(0 <= share <= 100 for share in shares) and 98 <= sum(shares) <= 102
    # The same command with more periods goes on from the saved training
    script.train("trial", periods=256, processes=1, every=64, per_process=1)
    progress = rows(folder)
    assert [int(row["periods"]) for row in progress] == [64, 128, 192, 256]
    assert float(progress[-1]["hours"]) >= float(progress[1]["hours"]) >= 0.0     # the hours add up
    assert all(0.0 < float(row["deviation"]) < 1.0 for row in progress)
    assert "Resuming" in capsys.readouterr().out
    # Nothing left to do
    script.train("trial", periods=256, processes=1, every=64, per_process=1)
    assert "Already trained" in capsys.readouterr().out and len(rows(folder)) == 4
    # A training resumes on as many robots
    with pytest.raises(SystemExit, match="1 robots"):
        script.train("trial", periods=512, processes=1, every=64, per_process=2)


def test_same_seed_same_training(short):
    for name in ("a", "b"):
        script.train(name, periods=128, processes=1, every=128, per_process=2)
    a, b = (script.PPO.load(short / name / "model.zip", device="cpu") for name in ("a", "b"))
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    assert np.array_equal(script.policy_of(a)(observation), script.policy_of(b)(observation))


def test_training_on_several_processes(short):
    script.train("parallel", periods=128, processes=2, every=128, per_process=2)     # 4 robots, 16 periods each
    assert [int(row["periods"]) for row in rows(short / "parallel")] == [128]
