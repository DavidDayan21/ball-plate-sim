"""Les deux physiques face à la théorie du roulement, et le modèle MuJoCo du robot."""
import math

import mujoco
import pytest

from controller.kinematics import Geometry, inverse_kinematics
from simulation.config import G, Config, PlateParams
from simulation.mujoco_physics import rolling_acceleration, build, plate_tilt, measured_restitution
from simulation.simple_physics import acceleration

K = 2 / 3


def test_simple_physics_hollow_ball():
    # ẍ = g·sin θ / (1 + k) = (3/5)·g·sin θ pour k = 2/3, axes indépendants
    ax, ay = acceleration(math.radians(2), 0.0, PlateParams())
    assert math.isclose(ax, 0.6 * G * math.sin(math.radians(2)), rel_tol=1e-12)
    assert ay == 0.0


@pytest.mark.parametrize("k", [2 / 3, 2 / 5])
def test_mujoco_matches_rolling_law(k):
    theta = math.radians(2)
    assert abs(rolling_acceleration(k, theta) / (G * math.sin(theta) / (1 + k)) - 1) < 0.01


def test_no_slip_condition():
    # La balle roule si μ ≥ k·tan θ / (1 + k), sinon elle glisse : a = g·(sin θ - μ·cos θ)
    th = math.radians(10)
    mu_min = K * math.tan(th) / (1 + K)
    rolls = rolling_acceleration(K, th, friction=1.3 * mu_min)
    slides = rolling_acceleration(K, th, friction=0.7 * mu_min)
    assert abs(rolls / (G * math.sin(th) / (1 + K)) - 1) < 0.01
    assert abs(slides / (G * (math.sin(th) - 0.7 * mu_min * math.cos(th))) - 1) < 0.02


def test_bounce_complies_with_ittf_rule():
    # Lâchée de 30,5 cm, une balle de ping-pong réglementaire remonte de 22 à 25 cm : e ∈ [0,85 ; 0,91]
    assert 0.85 <= measured_restitution(height=0.305) <= 0.91


def test_mujoco_robot_at_rest_plate_level():
    model, data = build(Config())
    for _ in range(1000):
        mujoco.mj_step(model, data)
    tx, ty = plate_tilt(data)
    assert abs(math.degrees(tx)) < 0.01 and abs(math.degrees(ty)) < 0.01
    assert abs(data.body("plate").xpos[2] - Geometry().height) < 0.001


def test_inverse_kinematics_validated_by_mujoco():
    # La chaîne fermée de MuJoCo (avec ses mouvements parasites) doit donner l'inclinaison voulue
    model, data = build(Config())
    data.ctrl[:] = inverse_kinematics(Geometry(), math.radians(5), math.radians(-3))
    for _ in range(2000):
        mujoco.mj_step(model, data)
    tx, ty = plate_tilt(data)
    assert abs(math.degrees(tx) - 5) < 0.05 and abs(math.degrees(ty) + 3) < 0.05
