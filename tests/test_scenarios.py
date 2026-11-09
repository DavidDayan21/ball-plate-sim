"""Complete closed-loop scenarios, from the simple physics to the MuJoCo + camera chain."""
import math
from dataclasses import replace

from simulation.loop import simulate
from simulation.config import Config, center
from simulation.metrics import theta_std, tracking_error_rms

SIMPLE_PLATE = Config(physics="simple", mechanism=False)
SIMPLE_ROBOT = Config(physics="simple", max_acceleration=math.inf)


def circle_error(cfg) -> float:
    return tracking_error_rms(simulate(cfg), 12, 20)


# --- Simple physics ---------------------------------------------------------------------------
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
    assert r.airborne_fraction < 0.01                      # the ball hardly ever leaves the plate


def test_ramps_suppress_bounces():
    # Elastic ball (e ≈ 0.88): without ramps, the jerks of the plate make it bounce
    assert simulate(Config(max_acceleration=math.inf)).airborne_fraction > 0.5
    assert simulate(Config()).airborne_fraction < 0.01


def test_robot_driven_by_camera():
    r = simulate(Config(sensor="camera"))
    assert not r.fallen and r.undetected == 0 and tracking_error_rms(r, 12, 20) < 0.008


def test_latency_degrades_tracking():
    uncompensated = Config(sensor="camera", latency_compensation=False)
    assert circle_error(replace(uncompensated, latency=5)) > 2 * circle_error(uncompensated)


def test_latency_compensation():
    # Compensation is on by default (part of the PID brain)
    assert Config().latency_compensation
    assert circle_error(Config(latency=5)) < 0.5 * circle_error(Config(latency=5, latency_compensation=False))


def test_thrown_ball_caught():
    r = simulate(Config(trajectory=center, duration=5.0, start_position=(-0.10, 0.0), start_velocity=(0.4, 0.0)))
    assert not r.fallen and math.hypot(r.x[-1], r.y[-1]) < 0.01


def test_ball_mass_has_no_effect():
    # ẍ = g·sin θ / (1 + k): the mass cancels out
    pm = Config().mujoco_params
    heavy = circle_error(Config(mujoco_params=replace(pm, ball_mass=0.027)))
    assert abs(heavy / circle_error(Config()) - 1) < 0.15


def test_dropped_ball_caught_by_the_aimed_bounce():
    # Dropped from 30.5 cm, 10 cm from the center (MuJoCo, synthetic sensor): caught when the brain aims
    # each bounce; reacting to the ball in the air like on the plate, the brain loses it
    cfg = Config(trajectory=center, start_position=(0.08, -0.06), start_height=0.305, duration=4.0)
    assert not simulate(cfg).fallen
    assert simulate(replace(cfg, airborne_height=math.inf)).fallen
