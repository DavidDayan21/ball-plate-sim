"""Bit-exact regression check: records the results of all simulations, then checks
that a change to the code does not alter them (refactorings must keep results bit-identical).

    python regression_check.py --record         BEFORE the change: writes the reference
    python regression_check.py --compare        AFTER: lists the differences (exit code 1 if there are any)
    python regression_check.py --compare --by-position
                                               same, when names have changed (translation, renaming):
                                               the results are matched in their recording order

Cases covered (seed 0, default settings), list frozen here because campaign.py will evolve:
    - all the simulations of the measurement campaign of 2026-11-03 (commit 7f2fd90);
    - its sweeps, reduced to two points around the limit (throws at ±10° and ±20°, pushes);
    - the physics validation (rolling, sliding, bounce);
    - a sandbox sequence: drop, throw caught, throw too fast;
    - added later: the simple physics with all its additions (full equation, servo lag).

Bit-exact identity is only meaningful on the same machine, with the same versions: as soon as the
ball bounces, Windows and Linux diverge. The reference therefore records the platform and the
versions, and the comparison warns if they have changed.

Reference: results/reference/reference.npz (ignored by git, specific to each machine).
"""
import argparse
from dataclasses import fields, replace
import datetime
from functools import partial
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from simulation.loop import Results, simulate
from simulation.config import Config, CameraParams, square, center
from simulation.parallel import execute
from simulation.mujoco_physics import rolling_acceleration, measured_restitution
from simulation.session import Session, sandbox_config

REFERENCE_FILE = Path(__file__).parent / "results" / "reference" / "reference.npz"
INFO_KEY = "__info__"        # reserved key: date and environment of the reference. The file format
                             # (this key and the JSON keys) is in English and must not change anymore.

# --- Reference cases ---------------------------------------------------------------------------
# Every setting that a case relies on is written explicitly, so that changing a default value of Config
# (as the brain or the robot improves) does not change the cases: they use the reference controller of the
# campaign (latency compensation off unless stated, error integrated everywhere, ball never considered in
# the air, no wait for the speed) and the motors of the first robot (single ramp of 50 rad/s²).
REFERENCE_CONTROLLER = {"latency_compensation": False, "integration_zone": math.inf,
                        "airborne_height": math.inf, "wait_for_speed": False, "fast_move": math.inf}
CAMERA = Config(sensor="camera", camera=CameraParams(width=160, image_height=120), **REFERENCE_CONTROLLER)
PM = CAMERA.mujoco_params


def _throw(tilt_deg: float, speed: float) -> Config:
    """Ball thrown from x = -10 cm toward +x, setpoint at the center."""
    return replace(CAMERA, theta_max=math.radians(tilt_deg), trajectory=center, duration=5.0,
                   start_position=(-0.10, 0.0), start_velocity=(speed, 0.0))


def _push(dv: float) -> Config:
    """Ball at rest at the center, pushed at t = 1 s."""
    return replace(CAMERA, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                   pushes=((1.0, 0.8 * dv, 0.6 * dv),))


SIMULATIONS = {
    "simple_plate": Config(physics="simple", mechanism=False, **REFERENCE_CONTROLLER),
    "simple_robot": Config(physics="simple", **REFERENCE_CONTROLLER),
    "mujoco_plate": Config(mechanism=False, **REFERENCE_CONTROLLER),
    "mujoco_robot": Config(**REFERENCE_CONTROLLER),
    "camera": CAMERA,
    "throw_10deg_0.5": _throw(10, 0.5),
    "throw_10deg_0.6": _throw(10, 0.6),
    "throw_20deg_0.6": _throw(20, 0.6),
    "throw_20deg_0.7": _throw(20, 0.7),
    "push_0.4": _push(0.4),
    "push_0.5": _push(0.5),
    "solid_ball": replace(CAMERA, mujoco_params=replace(PM, k_actual=2 / 5)),
    "heavy_ball": replace(CAMERA, mujoco_params=replace(PM, ball_mass=0.027)),
    "friction_0.2": replace(CAMERA, mujoco_params=replace(PM, friction=0.2)),
    "friction_0.02": replace(CAMERA, mujoco_params=replace(PM, friction=0.02)),
    "heavy_plate": replace(CAMERA, mujoco_params=replace(PM, plate_mass=0.5)),
    "latency_20ms_compensated": replace(CAMERA, latency_compensation=True),
    "latency_60ms": replace(CAMERA, latency=3),
    "latency_60ms_compensated": replace(CAMERA, latency=3, latency_compensation=True),
    "latency_100ms": replace(CAMERA, latency=5),
    "latency_100ms_compensated": replace(CAMERA, latency=5, latency_compensation=True),
    "square": replace(CAMERA, trajectory=square, duration=16.0, start_position=(0.05, 0.05)),
    "square_compensated": replace(CAMERA, trajectory=square, duration=16.0, start_position=(0.05, 0.05),
                              latency_compensation=True),
    # Added after the first reference
    "simple_robot_full_equation": Config(physics="simple", rolling_term=True, frame_terms=True,
                                         exact_gravity=True, servo_lag=True, **REFERENCE_CONTROLLER),
}


def extract(r: Results) -> dict:
    """All the fields of r as numpy arrays (the histories as float64)."""
    arrays = {}
    for f in fields(Results):
        value = getattr(r, f.name)
        arrays[f.name] = np.asarray(value, dtype=float) if isinstance(value, list) else np.asarray(value)
    return arrays


def simulation_results(cfg: Config) -> dict:
    return extract(simulate(cfg))


def _physics() -> dict:
    """Physics validation: section 1 of the campaign."""
    k = CAMERA.plate.k_inertia
    theta = math.radians(10)
    mu_min = k * math.tan(theta) / (1 + k)
    return {"rolling": np.asarray(rolling_acceleration(k, theta, friction=1.3 * mu_min)),
            "sliding": np.asarray(rolling_acceleration(k, theta, friction=0.7 * mu_min)),
            "restitution": np.asarray(measured_restitution())}


def _session() -> dict:
    """Sandbox: drop from 30.5 cm at the center, throw caught, throw too fast (fall)."""
    s = Session(replace(sandbox_config(), **REFERENCE_CONTROLLER), record=False)
    snapshots = []

    def advance(duration: float) -> None:
        for _ in range(round(duration / s.dt)):
            s.step()
        snapshots.append(np.asarray(list(s.curves), dtype=float))   # last 10 seconds

    advance(1.0)
    s.throw(0.0, 0.0, height=0.305)
    advance(6.0)
    s.throw(0.10, 180, speed=0.4)
    advance(4.0)
    s.throw(0.10, 180, speed=1.0)
    advance(3.0)
    rows = [_numeric_values(row) for row in s.close()]
    return {"curves": np.concatenate(snapshots),
            "summary": np.asarray([v for row in rows for v in row], dtype=float),
            "summary_sizes": np.asarray([len(row) for row in rows])}


def _numeric_values(row: dict) -> list:
    """Numeric values of a summary row, in order (None -> NaN, boolean -> 0 or 1).
    The texts (action, result, time...) are ignored: they do not depend on the computation, and they
    change if they are translated."""
    return [math.nan if v is None else float(v) for v in row.values() if not isinstance(v, str)]


CASES = {**{name: partial(simulation_results, cfg) for name, cfg in SIMULATIONS.items()},
       "physics": _physics, "session": _session}


def compute(name: str) -> dict:
    """Results of the case `name`: {field: array}."""
    return CASES[name]()


def compute_all(processes: int = None) -> dict:
    """Results of all the cases: {"case/field": array}."""
    names = list(CASES)
    arrays = {}
    for name, result in zip(names, execute(compute, names, processes)):
        arrays.update({f"{name}/{field_name}": value for field_name, value in result.items()})
    return arrays


# --- Recording and comparison --------------------------------------------------------------------
def environment() -> dict:
    """What bit-exact identity depends on: platform and versions."""
    import cv2
    import mujoco
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__,
            "mujoco": mujoco.__version__, "opencv": cv2.__version__}


def record(arrays: dict, file: Path = REFERENCE_FILE) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    info = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "environment": environment()}
    np.savez_compressed(file, **arrays, **{INFO_KEY: np.asarray(json.dumps(info, ensure_ascii=False))})


def load(file: Path = REFERENCE_FILE):
    """Returns (arrays, info) of a recorded reference."""
    with np.load(file) as data:
        arrays = {key: data[key] for key in data.files if key != INFO_KEY}
        info = json.loads(str(data[INFO_KEY]))
    return arrays, info


def _diff(a: np.ndarray, b: np.ndarray) -> str:
    """Description of the difference between two arrays; empty string if they are bit-identical."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return f"different type or shape ({a.dtype} {a.shape} -> {b.dtype} {b.shape})"
    if a.tobytes() == b.tobytes():
        return ""
    # Element-by-element comparison, on the bytes (0.0 and -0.0 differ, NaN is comparable)
    bytes_a = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    bytes_b = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(b.size, b.dtype.itemsize)
    differs = (bytes_a != bytes_b).any(axis=1)
    first = np.unravel_index(int(np.argmax(differs)), a.shape) if a.ndim else ()
    text = (f"{int(differs.sum())} value(s) differ out of {a.size}, "
             f"the first at index {tuple(int(i) for i in first)}")
    if np.issubdtype(a.dtype, np.number) or a.dtype == bool:
        d = np.abs(a.astype(float).reshape(-1) - b.astype(float).reshape(-1))[differs]
        text += f", max difference {np.max(d):.3g}"
    return text


def compare(reference: dict, current: dict) -> list:
    """Bit-exact differences between two sets {key: array}. Empty list: identical."""
    diffs = []
    for key in sorted(reference.keys() | current.keys()):
        if key not in current:
            diffs.append(f"{key}: missing from the current results")
        elif key not in reference:
            diffs.append(f"{key}: new, missing from the reference")
        elif text := _diff(reference[key], current[key]):
            diffs.append(f"{key}: {text}")
    return diffs


def compare_by_position(reference: dict, current: dict) -> list:
    """Like compare, but matches the arrays in their recording order, whatever their
    names. Used after a renaming: the order of the cases and fields must not change
    (a permutation would be reported as a difference, except between two identical arrays)."""
    diffs = []
    if len(reference) != len(current):
        diffs.append(f"different number of arrays: {len(reference)} -> {len(current)}")
    for (ref_key, a), (key, b) in zip(reference.items(), current.items()):
        if text := _diff(a, b):
            diffs.append(f"{ref_key} -> {key}: {text}")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-exact regression check of all the simulations")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record", action="store_true", help="writes the reference (before a change)")
    action.add_argument("--compare", action="store_true", help="compares to the reference (after the change)")
    parser.add_argument("--replace", action="store_true", help="with --record: overwrites the existing reference")
    parser.add_argument("--by-position", action="store_true",
                        help="with --compare: matches the results by their order, not by their name")
    parser.add_argument("--processes", type=int, default=None, help="number of processes (default: one per core)")
    args = parser.parse_args()

    if args.record and REFERENCE_FILE.exists() and not args.replace:
        sys.exit(f"A reference already exists ({REFERENCE_FILE}). Add --replace to overwrite it.")
    if args.compare and not REFERENCE_FILE.exists():
        sys.exit(f"No reference ({REFERENCE_FILE}): run --record first.")

    start = time.perf_counter()
    print(f"Computing {len(CASES)} cases...")
    current = compute_all(args.processes)
    print(f"Done in {time.perf_counter() - start:.0f} s.")

    if args.record:
        record(current)
        print(f"Reference written: {REFERENCE_FILE}")
        return

    reference, info = load()
    print(f"Reference from {info['date']}.")
    for key, value in environment().items():
        if info["environment"].get(key) != value:
            print(f"Warning, {key} differs: {info['environment'].get(key)} -> {value} "
                  "(bit-exact identity is no longer guaranteed).")
    if args.by_position:
        renamed_count = sum(a != b for a, b in zip(reference, current))
        print(f"Matching by position: {renamed_count} name(s) out of {len(reference)} have changed.")
        diffs = compare_by_position(reference, current)
    else:
        diffs = compare(reference, current)
    if not diffs:
        print(f"Bit-identical: {len(CASES)} cases out of {len(CASES)}.")
        return
    failed = sorted({e.split("/")[0] for e in diffs})
    print(f"DIFFERENCES in {len(failed)} cases out of {len(CASES)} ({', '.join(failed)}):")
    for e in diffs:
        print(f"  {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
