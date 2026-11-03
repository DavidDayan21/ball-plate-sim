"""Les deux physiques face à la théorie du roulement, et le modèle MuJoCo du robot."""
import math

import mujoco
import pytest

from controller.cinematique import Geometrie, cinematique_inverse
from simulation.config import G, Config, ParametresPlateau
from simulation.physique_mujoco import acceleration_roulement, construire, inclinaison_plateau
from simulation.physique_simple import acceleration

K = 2 / 3


def test_physique_simple_balle_creuse():
    # ẍ = g·sin θ / (1 + k) = (3/5)·g·sin θ pour k = 2/3, axes indépendants
    ax, ay = acceleration(math.radians(2), 0.0, ParametresPlateau())
    assert math.isclose(ax, 0.6 * G * math.sin(math.radians(2)), rel_tol=1e-12)
    assert ay == 0.0


@pytest.mark.parametrize("k", [2 / 3, 2 / 5])
def test_mujoco_retrouve_la_loi_du_roulement(k):
    theta = math.radians(2)
    assert abs(acceleration_roulement(k, theta) / (G * math.sin(theta) / (1 + k)) - 1) < 0.01


def test_condition_de_non_glissement():
    # La balle roule si μ ≥ k·tan θ / (1 + k), sinon elle glisse : a = g·(sin θ - μ·cos θ)
    th = math.radians(10)
    mu_min = K * math.tan(th) / (1 + K)
    roule = acceleration_roulement(K, th, frottement=1.3 * mu_min)
    glisse = acceleration_roulement(K, th, frottement=0.7 * mu_min)
    assert abs(roule / (G * math.sin(th) / (1 + K)) - 1) < 0.01
    assert abs(glisse / (G * (math.sin(th) - 0.7 * mu_min * math.cos(th))) - 1) < 0.02


def test_robot_mujoco_au_repos_plateau_horizontal():
    model, data = construire(Config())
    for _ in range(1000):
        mujoco.mj_step(model, data)
    tx, ty = inclinaison_plateau(data)
    assert abs(math.degrees(tx)) < 0.01 and abs(math.degrees(ty)) < 0.01
    assert abs(data.body("plateau").xpos[2] - Geometrie().hauteur) < 0.001


def test_cinematique_inverse_validee_par_mujoco():
    # La chaîne fermée de MuJoCo (avec ses mouvements parasites) doit donner l'inclinaison voulue
    model, data = construire(Config())
    data.ctrl[:] = cinematique_inverse(Geometrie(), math.radians(5), math.radians(-3))
    for _ in range(2000):
        mujoco.mj_step(model, data)
    tx, ty = inclinaison_plateau(data)
    assert abs(math.degrees(tx) - 5) < 0.05 and abs(math.degrees(ty) + 3) < 0.05
