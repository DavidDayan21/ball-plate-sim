"""Tests de la phase 3 : cinématique, moteurs, boucle complète du robot.

Lancement : python -m pytest
"""
import math
from dataclasses import replace

import pytest

from controller.cinematique import Geometrie, cinematique_directe, cinematique_inverse
from sim_simple.boucle_plateau import Config, erreur_suivi_rms
from sim_simple.boucle_plateau import simuler as simuler_plateau
from sim_simple.boucle_robot import ConfigRobot, simuler
from sim_simple.moteur import Moteur

G3 = Geometrie()


def test_repos_bras_horizontaux():
    assert all(abs(a) < 1e-12 for a in cinematique_inverse(G3, 0.0, 0.0))


@pytest.mark.parametrize("tx, ty", [(10, 0), (0, 10), (-7, 4), (5, -9), (3, 3)])
def test_aller_retour_inverse_directe(tx, ty):
    tx, ty = math.radians(tx), math.radians(ty)
    q = cinematique_directe(G3, cinematique_inverse(G3, tx, ty))
    assert abs(q[0] - tx) < 1e-9 and abs(q[1] - ty) < 1e-9 and abs(q[2] - G3.hauteur) < 1e-9


def test_rapport_de_transmission_petits_angles():
    # dα_i ≈ -(r/l1)·(cos φi·θx + sin φi·θy)
    tx, ty = math.radians(0.2), math.radians(-0.1)
    rapport = G3.rayon_attache / G3.longueur_bras
    for i, a in enumerate(cinematique_inverse(G3, tx, ty)):
        ux, uy = G3.direction(i)
        assert math.isclose(a, -rapport * (ux * tx + uy * ty), rel_tol=0.01, abs_tol=1e-6)


def test_pose_hors_portee():
    with pytest.raises(ValueError):
        cinematique_inverse(G3, math.radians(60), 0.0)


def test_moteur_vitesse_bornee_et_pas():
    m = Moteur(0.0, vitesse_max=2.0, resolution=0.01)
    m.demander(1.004)                         # arrondi au pas : 1,00
    for _ in range(100):
        m.avancer(0.001)                      # 0,1 s à 2 rad/s
    assert math.isclose(m.position, 0.2, rel_tol=1e-9)
    for _ in range(1000):
        m.avancer(0.001)
    assert math.isclose(m.position, 1.0, rel_tol=1e-9)


def test_robot_sans_moteurs_identique_a_la_phase_2():
    r2 = simuler_plateau(Config())
    r3 = simuler(ConfigRobot(moteurs=False))
    assert r2.x == r3.x and r2.y == r3.y


def test_robot_complet():
    r = simuler(ConfigRobot())
    assert not r.tombee
    assert r.poses_hors_portee == 0
    assert erreur_suivi_rms(r, 12, 20) < 0.006


def test_moteurs_lents_degradent_le_suivi():
    rapide = erreur_suivi_rms(simuler(ConfigRobot()), 12, 20)
    lent = erreur_suivi_rms(simuler(replace(ConfigRobot(), vitesse_max=0.5)), 12, 20)
    assert lent > 1.5 * rapide
