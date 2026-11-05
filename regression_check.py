"""Non-régression au bit près : enregistre les résultats de toutes les simulations, puis vérifie
qu'une modification du code ne les change pas (règle 5 du projet).

    python regression_check.py --record         AVANT la modification : écrit la référence
    python regression_check.py --compare        APRÈS : liste les écarts (code de sortie 1 s'il y en a)
    python regression_check.py --compare --by-position
                                               idem quand des noms ont changé (traduction, renommage) :
                                               les résultats sont appariés dans leur ordre d'enregistrement

Cas couverts (graine 0, réglages par défaut), liste figée ici car campaign.py va évoluer :
    - toutes les simulations de la campagne de mesures du 03/11/2026 (commit 7f2fd90) ;
    - ses balayages, réduits à deux points autour de la limite (lancers à ±10° et ±20°, poussées) ;
    - la validation de la physique (roulement, glissement, rebond) ;
    - un enchaînement du bac à sable : lâcher, lancer rattrapé, lancer trop rapide.

L'identité au bit près n'a de sens que sur une même machine, avec les mêmes versions : dès que la
balle rebondit, Windows et Linux divergent. La référence enregistre donc la plateforme et les
versions, et la comparaison prévient si elles ont changé.

Référence : results/reference/reference.npz (ignorée par git, propre à chaque machine).
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
from simulation.session import Session

REFERENCE_FILE = Path(__file__).parent / "results" / "reference" / "reference.npz"
INFO_KEY = "__info__"           # clé réservée : date et environnement de la référence. Le format du fichier
                             # (cette clé et les clés JSON) est en anglais et ne doit plus changer.

# --- Cas de référence --------------------------------------------------------------------------
CAMERA = Config(sensor="camera", camera=CameraParams(width=160, image_height=120))
PM = CAMERA.mujoco_params


def _throw(tilt_deg: float, speed: float) -> Config:
    """Balle lancée depuis x = -10 cm vers +x, consigne au centre."""
    return replace(CAMERA, theta_max=math.radians(tilt_deg), trajectory=center, duration=5.0,
                   start_position=(-0.10, 0.0), start_velocity=(speed, 0.0))


def _push(dv: float) -> Config:
    """Balle au repos au centre, poussée à t = 1 s."""
    return replace(CAMERA, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                   pushes=((1.0, 0.8 * dv, 0.6 * dv),))


SIMULATIONS = {
    "simple_plate": Config(physics="simple", mechanism=False),
    "simple_robot": Config(physics="simple"),
    "mujoco_plate": Config(mechanism=False),
    "mujoco_robot": Config(),
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
}


def extract(r: Results) -> dict:
    """Tous les champs de r en tableaux numpy (les historiques en float64)."""
    arrays = {}
    for f in fields(Results):
        value = getattr(r, f.name)
        arrays[f.name] = np.asarray(value, dtype=float) if isinstance(value, list) else np.asarray(value)
    return arrays


def simulation_results(cfg: Config) -> dict:
    return extract(simulate(cfg))


def _physics() -> dict:
    """Validation de la physique : section 1 de la campagne."""
    k = CAMERA.plate.k_inertia
    theta = math.radians(10)
    mu_min = k * math.tan(theta) / (1 + k)
    return {"rolling": np.asarray(rolling_acceleration(k, theta, friction=1.3 * mu_min)),
            "sliding": np.asarray(rolling_acceleration(k, theta, friction=0.7 * mu_min)),
            "restitution": np.asarray(measured_restitution())}


def _session() -> dict:
    """Bac à sable : lâcher de 30,5 cm au centre, lancer rattrapé, lancer trop rapide (chute)."""
    s = Session(record=False)
    snapshots = []

    def advance(duration: float) -> None:
        for _ in range(round(duration / s.dt)):
            s.step()
        snapshots.append(np.asarray(list(s.curves), dtype=float))   # 10 dernières secondes

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
    """Valeurs numériques d'une ligne du résumé, dans l'ordre (None -> NaN, booléen -> 0 ou 1).
    Les textes (action, résultat, heure...) sont ignorés : ils ne dépendent pas du calcul, et ils
    changeront si on les traduit."""
    return [math.nan if v is None else float(v) for v in row.values() if not isinstance(v, str)]


CASES = {**{name: partial(simulation_results, cfg) for name, cfg in SIMULATIONS.items()},
       "physics": _physics, "session": _session}


def compute(name: str) -> dict:
    """Résultats du cas `nom` : {champ: tableau}."""
    return CASES[name]()


def compute_all(processes: int = None) -> dict:
    """Résultats de tous les cas : {"cas/champ": tableau}."""
    names = list(CASES)
    arrays = {}
    for name, result in zip(names, execute(compute, names, processes)):
        arrays.update({f"{name}/{field_name}": value for field_name, value in result.items()})
    return arrays


# --- Enregistrement et comparaison ---------------------------------------------------------------
def environment() -> dict:
    """Ce dont dépend l'identité au bit près : plateforme et versions."""
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
    """Renvoie (tableaux, infos) d'une référence enregistrée."""
    with np.load(file) as data:
        arrays = {key: data[key] for key in data.files if key != INFO_KEY}
        info = json.loads(str(data[INFO_KEY]))
    return arrays, info


def _diff(a: np.ndarray, b: np.ndarray) -> str:
    """Description de la différence entre deux tableaux ; chaîne vide s'ils sont identiques au bit près."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return f"type ou forme différents ({a.dtype} {a.shape} -> {b.dtype} {b.shape})"
    if a.tobytes() == b.tobytes():
        return ""
    # Comparaison élément par élément, sur les octets (0,0 et -0,0 diffèrent, NaN est comparable)
    bytes_a = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    bytes_b = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(b.size, b.dtype.itemsize)
    differs = (bytes_a != bytes_b).any(axis=1)
    first = np.unravel_index(int(np.argmax(differs)), a.shape) if a.ndim else ()
    text = (f"{int(differs.sum())} valeur(s) différente(s) sur {a.size}, "
             f"la première à l'indice {tuple(int(i) for i in first)}")
    if np.issubdtype(a.dtype, np.number) or a.dtype == bool:
        d = np.abs(a.astype(float).reshape(-1) - b.astype(float).reshape(-1))[differs]
        text += f", écart max {np.max(d):.3g}"
    return text


def compare(reference: dict, current: dict) -> list:
    """Écarts au bit près entre deux jeux {clé: tableau}. Liste vide : identiques."""
    diffs = []
    for key in sorted(reference.keys() | current.keys()):
        if key not in current:
            diffs.append(f"{key} : absent des résultats actuels")
        elif key not in reference:
            diffs.append(f"{key} : nouveau, absent de la référence")
        elif text := _diff(reference[key], current[key]):
            diffs.append(f"{key} : {text}")
    return diffs


def compare_by_position(reference: dict, current: dict) -> list:
    """Comme comparer, mais apparie les tableaux dans leur ordre d'enregistrement, quels que soient
    leurs noms. Sert après un renommage : l'ordre des cas et des champs ne doit pas changer
    (une permutation serait signalée comme un écart, sauf entre deux tableaux identiques)."""
    diffs = []
    if len(reference) != len(current):
        diffs.append(f"nombre de tableaux différent : {len(reference)} -> {len(current)}")
    for (ref_key, a), (key, b) in zip(reference.items(), current.items()):
        if text := _diff(a, b):
            diffs.append(f"{ref_key} -> {key} : {text}")
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-régression au bit près de toutes les simulations")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record", action="store_true", help="écrit la référence (avant une modification)")
    action.add_argument("--compare", action="store_true", help="compare à la référence (après la modification)")
    parser.add_argument("--replace", action="store_true", help="avec --record : écrase la référence existante")
    parser.add_argument("--by-position", action="store_true",
                        help="avec --compare : apparie les résultats par leur ordre, pas par leur nom")
    parser.add_argument("--processes", type=int, default=None, help="nombre de processus (défaut : un par cœur)")
    args = parser.parse_args()

    if args.record and REFERENCE_FILE.exists() and not args.replace:
        sys.exit(f"Une référence existe déjà ({REFERENCE_FILE}). Ajoute --replace pour l'écraser.")
    if args.compare and not REFERENCE_FILE.exists():
        sys.exit(f"Aucune référence ({REFERENCE_FILE}) : lance d'abord --record.")

    start = time.perf_counter()
    print(f"Calcul de {len(CASES)} cas...")
    current = compute_all(args.processes)
    print(f"Terminé en {time.perf_counter() - start:.0f} s.")

    if args.record:
        record(current)
        print(f"Référence écrite : {REFERENCE_FILE}")
        return

    reference, info = load()
    print(f"Référence du {info['date']}.")
    for key, value in environment().items():
        if info["environment"].get(key) != value:
            print(f"Attention, {key} différent : {info['environment'].get(key)} -> {value} "
                  "(l'identité au bit près n'est plus garantie).")
    if args.by_position:
        renamed_count = sum(a != b for a, b in zip(reference, current))
        print(f"Appariement par position : {renamed_count} nom(s) sur {len(reference)} ont changé.")
        diffs = compare_by_position(reference, current)
    else:
        diffs = compare(reference, current)
    if not diffs:
        print(f"Identique au bit près : {len(CASES)} cas sur {len(CASES)}.")
        return
    failed = sorted({e.split("/")[0] for e in diffs})
    print(f"ÉCARTS dans {len(failed)} cas sur {len(CASES)} ({', '.join(failed)}) :")
    for e in diffs:
        print(f"  {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
