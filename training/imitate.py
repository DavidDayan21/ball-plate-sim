"""Imitation: the policy of the AI brain learns to answer like the PID brain, its teacher.

    python -m training.imitate --name pupil

Needs PyTorch (pip install -r requirements-train.txt). The pupil is the network of controller/network.py behind
the observation of controller/policy.py; the teacher is the PID brain with the settings of each episode. The
episodes are those of the training by reinforcement (training/environment.py).

A lesson is an episode where both see the same measurements: at every control period, the observation of the
pupil and the command of the teacher are recorded. The network then learns to give that command for that
observation (a regression). Rounds:
    0       the teacher drives. Half of its episodes with a random deviation added to its commands, so that the
            lessons also show how it recovers from commands that are not its own (the error printed is in
            degrees of tilt, for a maximum tilt of 10°)
    1, 2…   the pupil drives, the teacher only tells what it would have done (it is told the commands really
            sent): the pupil learns to recover from its own errors, which it never meets behind the teacher.
            All the lessons since round 0 are learned again
The policy of the last round is the brain: it drives alone, without the teacher.

After every round, the pupil alone is compared with the teacher alone: on the reference scenario of the campaign
(training/evaluation.py), and on VALIDATION episodes that are never learned, the same for both (share of the
episodes where the ball falls, mean distance to the setpoint).

Written in results/training/<name>/ (ignored by git): policy.npz, the network after the last round finished.
"""
import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from controller.network import Network
from controller.policy import OBSERVATION_SIZE, PolicyController
from simulation.config import Config
from simulation.loop import Simulation, pid_brain, policy_brain
from simulation.parallel import execute
from training.environment import draw_episode
from training.evaluation import evaluate

FOLDER = Path("results") / "training"
HIDDEN = (128, 128)                  # neurons of the two hidden layers, as in training/train.py
VALIDATION = 300                     # episodes
DEVIATION = 0.1                      # round 0: standard deviation added to the commands of the teacher, in units
                                     # of the maximum tilt (1°)
BATCH = 4096
LEARNING_RATE = (1e-3, 1e-5)         # at the start and at the end of each round


class Lesson:
    """Brain of an episode of imitation: the teacher (PID brain) and the pupil (policy) see the same measurements,
    one of them drives; records the observation of the pupil and the command of the teacher, in units of the
    maximum tilt, at every control period.

    policy: the pupil drives with it; None: the teacher drives. deviation: standard deviation of a random
    deviation added to the command sent (units of the maximum tilt), drawn with rng."""
    model_gain = math.nan

    def __init__(self, cfg: Config, policy=None, deviation: float = 0.0, rng: random.Random = None):
        self.teacher = pid_brain(cfg)
        self.pupil = PolicyController(policy, cfg.theta_max, cfg.latency)
        self.deviation, self.rng = deviation, rng
        self.observations, self.targets = [], []

    def compute(self, setpoint, measurement, setpoint_acc, dt: float):
        observation = self.pupil.observe(setpoint, measurement, setpoint_acc, dt)
        advice = self.teacher.compute(setpoint, measurement, setpoint_acc, dt)
        target = (advice[0] / self.pupil.theta_max, advice[1] / self.pupil.theta_max)
        self.observations.append(observation)
        self.targets.append(target)
        action = target if self.pupil.policy is None else self.pupil.policy(observation)
        if self.deviation > 0:
            action = (action[0] + self.rng.gauss(0, self.deviation), action[1] + self.rng.gauss(0, self.deviation))
        command = self.pupil.act(action)                # remembered by the pupil
        self.teacher.applied(command)
        return command


def play(job):
    """Lessons of several episodes. job = (seeds, layers, deviation, base): one episode per seed, drawn from the
    Config `base`; layers: weights of the network of the pupil, which drives (None: the teacher drives).
    Returns (observations, targets, falls, mean distance to the setpoint in m over the periods played)."""
    seeds, layers, deviation, base = job
    policy = None if layers is None else Network(layers)
    observations, targets, falls, distance, periods = [], [], 0, 0.0, 0
    for seed in seeds:
        rng = random.Random(seed)
        sim = Simulation(draw_episode(rng, base), lambda cfg: Lesson(cfg, policy, deviation, rng))
        for _ in range(round(sim.cfg.duration / sim.cfg.dt_physics) + 1):
            if not sim.step():
                break
        r = sim.finish()
        observations += sim.brain.observations
        targets += sim.brain.targets
        falls += r.fallen
        distance += sum(math.hypot(x - xr, y - yr) for x, y, xr, yr in zip(r.x, r.y, r.x_ref, r.y_ref))
        periods += len(r.x)
    return (np.array(observations, dtype=np.float32).reshape(-1, OBSERVATION_SIZE),
            np.array(targets, dtype=np.float32).reshape(-1, 2), falls, distance / max(1, periods))


def lessons(seeds, layers, deviation: float, processes: int, base: Config = None):
    """play() over all the seeds, spread over the processes. Returns (observations, targets, share of the
    episodes where the ball fell, mean distance to the setpoint)."""
    seeds = list(seeds)
    chunk = max(1, math.ceil(len(seeds) / (4 * processes)))
    jobs = [(seeds[k:k + chunk], layers, deviation, base) for k in range(0, len(seeds), chunk)]
    out = execute(play, jobs, processes)
    weights = [len(job[0]) for job in jobs]
    return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
            sum(o[2] for o in out) / len(seeds), float(np.average([o[3] for o in out], weights=weights)))


class Pupil:
    """The network being learned (PyTorch). Its inputs are centered and scaled with the statistics of the first
    lessons; layers() gives the weights of the same function for controller/network.py."""

    def __init__(self, observations: np.ndarray, seed: int = 0):
        torch.manual_seed(seed)
        self.mean = torch.as_tensor(observations.mean(axis=0))
        self.scale = torch.as_tensor(np.maximum(observations.std(axis=0), 0.01))
        sizes = (OBSERVATION_SIZE, *HIDDEN, 2)
        modules = []
        for inputs, outputs in zip(sizes, sizes[1:]):
            modules += [torch.nn.Linear(inputs, outputs), torch.nn.Tanh()]
        self.net = torch.nn.Sequential(*modules[:-1])

    def fit(self, observations: np.ndarray, targets: np.ndarray, epochs: int) -> float:
        """Learns to give the targets for the observations. Returns the root mean square error over the last
        epoch, in units of the maximum tilt."""
        x = (torch.as_tensor(observations) - self.mean) / self.scale
        y = torch.as_tensor(targets)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE[0])
        batches = math.ceil(len(x) / BATCH)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * batches, LEARNING_RATE[1])
        for _ in range(epochs):
            order, total = torch.randperm(len(x)), 0.0
            for k in range(batches):
                rows = order[k * BATCH:(k + 1) * BATCH]
                loss = torch.nn.functional.mse_loss(self.net(x[rows]), y[rows])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                total += loss.item() * len(rows)
        return math.sqrt(total / len(x))

    def layers(self) -> list:
        linear = [m for m in self.net if isinstance(m, torch.nn.Linear)]
        layers = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in linear]
        w, b = layers[0]                                # W·(x - mean) / scale + b = (W / scale)·x + b - (W / scale)·mean
        w = w / self.scale.double().numpy()
        layers[0] = (w, b - w @ self.mean.double().numpy())
        return layers


def report(who: str, layers, processes: int, base: Config, seed: int) -> None:
    """Prints how the brain alone does (layers: the pupil with these weights; None: the teacher)."""
    marks = evaluate(pid_brain if layers is None else policy_brain(Network(layers)))
    seeds = range(seed * 10_000_000 + 9_000_000, seed * 10_000_000 + 9_000_000 + VALIDATION)
    _, _, falls, distance = lessons(seeds, layers, 0.0, processes, base)
    print(f"  {who} alone | return {marks['return_20ms_mm']:.2f} mm, circle {marks['circle_20ms_mm']:.2f} mm, tremor "
          f"{marks['tremor_20ms_deg']:.2f}°; at 60 ms: {marks['return_60ms_mm']:.2f} mm, "
          f"{marks['circle_60ms_mm']:.2f} mm, {marks['tremor_60ms_deg']:.2f}° | validation: falls "
          f"{100 * falls:.1f} %, mean distance {1000 * distance:.2f} mm", flush=True)


def imitate(name: str, rounds: int, episodes: int, epochs: int, processes: int, base: Config = None,
            seed: int = 0) -> Network:
    """Rounds of `episodes` lessons each (see the top of the file), learned during `epochs` epochs per round."""
    folder = FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(processes)
    start, first = time.time(), seed * 10_000_000
    print(f"Imitation {folder}: {rounds} rounds of {episodes} episodes on {processes} processes.")
    report("teacher", None, processes, base, seed)
    half = episodes // 2
    plain = lessons(range(first, first + half), None, 0.0, processes, base)
    disturbed = lessons(range(first + half, first + episodes), None, DEVIATION, processes, base)
    observations, targets = (np.concatenate(pair) for pair in zip(plain[:2], disturbed[:2]))
    pupil, network = Pupil(observations, seed), None
    for k in range(rounds):
        if k > 0:                                       # the pupil drives
            seeds = range(first + k * episodes, first + (k + 1) * episodes)
            new = lessons(seeds, network.layers, 0.0, processes, base)
            observations, targets = np.concatenate((observations, new[0])), np.concatenate((targets, new[1]))
        error = pupil.fit(observations, targets, epochs)
        network = Network(pupil.layers())
        network.save(folder / "policy.npz")
        print(f"round {k}: {len(observations)} periods of lessons learned, error {10 * error:.3f}° rms "
              f"({(time.time() - start) / 60:.0f} min)", flush=True)
        report("pupil  ", network.layers, processes, base, seed)
    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation of the PID brain by the policy of the AI brain")
    parser.add_argument("--name", required=True, help="folder of the policy in results/training/")
    parser.add_argument("--rounds", type=int, default=6, help="rounds of lessons, the first with the teacher driving")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes of 10 s per round")
    parser.add_argument("--epochs", type=int, default=30, help="passes over all the lessons at every round")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    imitate(args.name, args.rounds, args.episodes, args.epochs, args.processes)
