"""Scénarios complets en boucle fermée, de la physique simple à la chaîne MuJoCo + caméra."""
import math
from dataclasses import replace

from simulation.loop import simulate
from simulation.config import Config, center
from simulation.metrics import theta_std, tracking_error_rms

SIMPLE_PLATE = Config(physics="simple", mechanism=False)
SIMPLE_ROBOT = Config(physics="simple", max_acceleration=math.inf)


def circle_error(cfg) -> float:
    return tracking_error_rms(simulate(cfg), 12, 20)


# --- Physique simple --------------------------------------------------------------------------
def test_derivative_filter_reduces_noise():
    filtered = theta_std(simulate(SIMPLE_PLATE), 2.5, 4)
    raw = theta_std(simulate(replace(SIMPLE_PLATE, tau_d=0.0)), 2.5, 4)
    assert filtered < 0.5 * raw


def test_feedforward_improves_tracking():
    assert circle_error(SIMPLE_PLATE) < 0.2 * circle_error(replace(SIMPLE_PLATE, feedforward=False))


def test_simple_plate():
    r = simulate(SIMPLE_PLATE)
    assert not r.fallen and tracking_error_rms(r, 12, 20) < 0.006


def test_simple_robot():
    r = simulate(SIMPLE_ROBOT)
    assert not r.fallen and r.unreachable_poses == 0 and tracking_error_rms(r, 12, 20) < 0.006


def test_slow_motors_degrade_tracking():
    assert circle_error(replace(SIMPLE_ROBOT, max_speed=0.5)) > 1.5 * circle_error(SIMPLE_ROBOT)


# --- MuJoCo -------------------------------------------------------------------------------------
def test_mujoco_angle_driven_plate():
    r = simulate(Config(mechanism=False))
    assert not r.fallen and tracking_error_rms(r, 12, 20) < 0.008


def test_mujoco_robot_with_ramps():
    r = simulate(Config())
    assert not r.fallen and r.unreachable_poses == 0
    assert tracking_error_rms(r, 12, 20) < 0.008
    assert r.airborne_fraction < 0.01                      # la balle ne décolle quasiment pas


def test_ramps_suppress_bounces():
    # Balle élastique (e ≈ 0,88) : sans rampes, les à-coups du plateau la font rebondir
    assert simulate(Config(max_acceleration=math.inf)).airborne_fraction > 0.5
    assert simulate(Config()).airborne_fraction < 0.01


def test_robot_driven_by_camera():
    r = simulate(Config(sensor="camera"))
    assert not r.fallen and r.undetected == 0 and tracking_error_rms(r, 12, 20) < 0.008


def test_latency_degrades_tracking():
    assert circle_error(Config(sensor="camera", latency=5)) > 2 * circle_error(Config(sensor="camera"))


def test_latency_compensation():
    assert circle_error(Config(latency=5, latency_compensation=True)) < 0.5 * circle_error(Config(latency=5))


def test_thrown_ball_caught():
    r = simulate(Config(trajectory=center, duration=5.0, start_position=(-0.10, 0.0), start_velocity=(0.4, 0.0)))
    assert not r.fallen and math.hypot(r.x[-1], r.y[-1]) < 0.01


def test_ball_mass_has_no_effect():
    # ẍ = g·sin θ / (1 + k) : la masse se simplifie
    pm = Config().mujoco_params
    heavy = circle_error(Config(mujoco_params=replace(pm, ball_mass=0.027)))
    assert abs(heavy / circle_error(Config()) - 1) < 0.15
