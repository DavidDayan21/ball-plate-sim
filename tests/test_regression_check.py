"""Outil de non-régression au bit près (regression_check.py) et exécution en parallèle."""
from dataclasses import fields, replace
import math

import numpy as np

import regression_check as rc
from simulation.loop import Results, simulate
from simulation.config import Config
from simulation.parallel import execute

SHORT = Config(physics="simple", mechanism=False, duration=1.0)


def test_extract_all_fields():
    t = rc.extract(simulate(SHORT))
    assert set(t) == {f.name for f in fields(Results)}
    assert t["x"].shape == (51,) and t["actual_tilt"].shape == (51, 2)   # 1 s à 50 Hz
    assert t["fallen"].dtype == bool and t["fallen"].shape == ()


def test_compare_detects_single_bit():
    reference = rc.extract(simulate(SHORT))
    modified = {k: v.copy() for k, v in reference.items()}
    assert rc.compare(reference, modified) == []
    modified["x"][10] = np.nextafter(modified["x"][10], np.inf)          # dernier bit de la mantisse
    diffs = rc.compare(reference, modified)
    assert len(diffs) == 1 and diffs[0].startswith("x : 1 valeur(s) différente(s) sur 51, la première à l'indice (10,)")


def test_compare_keys_shapes_and_signed_zero():
    a = {"c/zero": np.array([0.0]), "c/length": np.zeros(2), "c/removed": np.zeros(1)}
    b = {"c/zero": np.array([-0.0]), "c/length": np.zeros(3), "c/added": np.zeros(1)}
    assert [e.split(" :")[0] for e in rc.compare(a, b)] == ["c/added", "c/length", "c/removed", "c/zero"]


def test_compare_by_position_ignores_names():
    a = {"case/x": np.arange(3.0), "case/y": np.ones(2)}
    renamed = {"renamed/x": np.arange(3.0), "renamed/y": np.ones(2)}
    assert rc.compare_by_position(a, renamed) == []
    permuted = {"renamed/y": np.ones(2), "renamed/x": np.arange(3.0)}
    assert len(rc.compare_by_position(a, permuted)) == 2          # un ordre changé est signalé
    assert rc.compare_by_position(a, {"renamed/x": np.arange(3.0)})[0].startswith("nombre de tableaux")


def test_session_summary_without_text():
    row = {"number": 2, "action": "drop", "height_cm": 30.5, "stabilized_in_s": None, "compensation": True}
    values = rc._numeric_values(row)
    assert values[:2] == [2.0, 30.5] and math.isnan(values[2]) and values[3] == 1.0 and len(values) == 4


def test_record_then_load(tmp_path):
    arrays = {f"case/{k}": v for k, v in rc.extract(simulate(SHORT)).items()}
    arrays["session/summary"] = np.asarray(["{'result': 'fallen'}"])
    rc.record(arrays, tmp_path / "reference.npz")
    reloaded, info = rc.load(tmp_path / "reference.npz")
    assert rc.compare(arrays, reloaded) == []
    assert info["environment"] == rc.environment()


def test_parallel_bit_identical():
    # Processus redémarrés (comme sous Windows) : même résultat qu'à la suite, caméra comprise
    assert execute(math.sqrt, [1.0, 4.0, 9.0], processes=2) == [1.0, 2.0, 3.0]
    cfg = replace(rc.CAMERA, duration=0.5)
    direct = rc.simulation_results(cfg)
    for result in execute(rc.simulation_results, [cfg, cfg], processes=2):
        assert rc.compare(direct, result) == []
