"""Tests de la phase 4 : physique MuJoCo, modèle du robot, rampes des moteurs.

Lancement : python -m pytest
"""
import math

import mujoco
import pytest

from controller.cinematique import Geometrie, cinematique_inverse
from sim_mujoco.boucle_mujoco import (ConfigMujoco, acceleration_roulement, construire,
                                      inclinaison_plateau, simuler)
from sim_simple.boucle_plateau import erreur_suivi_rms
from sim_simple.moteur import Moteur


@pytest.mark.parametrize("k", [2 / 3, 2 / 5])
def test_mujoco_retrouve_la_loi_du_roulement(k):
    theta = math.radians(2)
    theorie = 9.81 * math.sin(theta) / (1 + k)
    assert abs(acceleration_roulement(k, theta) / theorie - 1) < 0.01


def test_robot_mujoco_au_repos_plateau_horizontal():
    model, data = construire(ConfigMujoco())
    for _ in range(1000):
        mujoco.mj_step(model, data)
    tx, ty = inclinaison_plateau(data)
    assert abs(math.degrees(tx)) < 0.01 and abs(math.degrees(ty)) < 0.01
    assert abs(data.body("plateau").xpos[2] - Geometrie().hauteur) < 0.001


def test_cinematique_inverse_validee_par_mujoco():
    # La chaîne fermée de MuJoCo (avec ses mouvements parasites) doit donner l'inclinaison voulue
    model, data = construire(ConfigMujoco())
    data.ctrl[:] = cinematique_inverse(Geometrie(), math.radians(5), math.radians(-3))
    for _ in range(2000):
        mujoco.mj_step(model, data)
    tx, ty = inclinaison_plateau(data)
    assert abs(math.degrees(tx) - 5) < 0.05 and abs(math.degrees(ty) + 3) < 0.05


def test_moteur_rampe_acceleration_bornee():
    m = Moteur(0.0, vitesse_max=10.0, resolution=1e-4, acceleration_max=50.0)
    m.demander(1.0)
    vitesses = []
    for _ in range(600):
        m.avancer(0.001)
        vitesses.append(m.vitesse)
    sauts = [abs(b - a) / 0.001 for a, b in zip(vitesses, vitesses[1:])]
    assert max(sauts) <= 50.0 + 1e-6
    assert abs(m.position - 1.0) < 1e-3


def test_plateau_mujoco_scenario():
    r = simuler(ConfigMujoco(mecanisme=False))
    assert not r.tombee
    assert erreur_suivi_rms(r, 12, 20) < 0.008


def test_robot_mujoco_scenario_avec_rampes():
    cfg = ConfigMujoco()
    r = simuler(cfg)
    assert not r.tombee
    assert r.poses_hors_portee == 0
    assert erreur_suivi_rms(r, 12, 20) < 0.008
    assert r.part_sans_contact < 0.01                      # la balle ne décolle quasiment plus


def test_rampes_suppriment_les_rebonds():
    sans = simuler(ConfigMujoco(acceleration_max=math.inf)).part_sans_contact
    avec = simuler(ConfigMujoco()).part_sans_contact
    assert sans > 10 * avec
