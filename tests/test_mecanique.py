"""Cinématique du mécanisme à 3 bras et modèle des moteurs."""
import math

import pytest

from controller.cinematique import Geometrie, cinematique_directe, cinematique_inverse
from simulation.moteur import Moteur

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


def test_moteur_rampe_acceleration_bornee():
    m = Moteur(0.0, vitesse_max=10.0, resolution=1e-4, acceleration_max=50.0)
    m.demander(1.0)
    vitesses = []
    for _ in range(600):
        m.avancer(0.001)
        vitesses.append(m.vitesse)
    assert max(abs(b - a) / 0.001 for a, b in zip(vitesses, vitesses[1:])) <= 50.0 + 1e-6
    assert abs(m.position - 1.0) < 1e-3
