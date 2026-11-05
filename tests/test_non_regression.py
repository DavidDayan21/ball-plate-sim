"""Outil de non-régression au bit près (non_regression.py) et exécution en parallèle."""
from dataclasses import fields, replace
import math

import numpy as np

import non_regression as nr
from simulation.boucle import Resultats, simuler
from simulation.config import Config
from simulation.parallele import executer

COURT = Config(physique="simple", mecanisme=False, duree=1.0)


def test_extraire_tous_les_champs():
    t = nr.extraire(simuler(COURT))
    assert set(t) == {f.name for f in fields(Resultats)}
    assert t["x"].shape == (51,) and t["inclinaison_reelle"].shape == (51, 2)   # 1 s à 50 Hz
    assert t["tombee"].dtype == bool and t["tombee"].shape == ()


def test_comparer_detecte_un_seul_bit():
    reference = nr.extraire(simuler(COURT))
    copie = {k: v.copy() for k, v in reference.items()}
    assert nr.comparer(reference, copie) == []
    copie["x"][10] = np.nextafter(copie["x"][10], np.inf)          # dernier bit de la mantisse
    ecarts = nr.comparer(reference, copie)
    assert len(ecarts) == 1 and ecarts[0].startswith("x : 1 valeur(s) différente(s) sur 51, la première à l'indice (10,)")


def test_comparer_cles_formes_et_zero_signe():
    a = {"c/zero": np.array([0.0]), "c/forme": np.zeros(2), "c/retire": np.zeros(1)}
    b = {"c/zero": np.array([-0.0]), "c/forme": np.zeros(3), "c/ajoute": np.zeros(1)}
    assert [e.split(" :")[0] for e in nr.comparer(a, b)] == ["c/ajoute", "c/forme", "c/retire", "c/zero"]


def test_enregistrer_puis_charger(tmp_path):
    tableaux = {f"cas/{k}": v for k, v in nr.extraire(simuler(COURT)).items()}
    tableaux["session/resume"] = np.asarray(["{'resultat': 'tombée'}"])
    nr.enregistrer(tableaux, tmp_path / "reference.npz")
    relus, infos = nr.charger(tmp_path / "reference.npz")
    assert nr.comparer(tableaux, relus) == []
    assert infos["environnement"] == nr.environnement()


def test_parallele_identique_au_bit_pres():
    # Processus redémarrés (comme sous Windows) : même résultat qu'à la suite, caméra comprise
    assert executer(math.sqrt, [1.0, 4.0, 9.0], processus=2) == [1.0, 2.0, 3.0]
    cfg = replace(nr.CAMERA, duree=0.5)
    direct = nr.resultats_simulation(cfg)
    for resultat in executer(nr.resultats_simulation, [cfg, cfg], processus=2):
        assert nr.comparer(direct, resultat) == []
