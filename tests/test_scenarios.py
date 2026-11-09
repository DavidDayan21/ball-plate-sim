"""Complete closed-loop scenarios, from the simple physics to the MuJoCo + camera chain."""
import math
from dataclasses import replace
import statistics

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
    # Without motors, the plate jumps to each new command: the ball bounces 20 to 50 % of the time and its
    # path is chaotic (a start shifted by 1 nm changes the error from 1.2 to 4 mm or more, and the last bits
    # of the computations differ between machines). Judged on the median of 5 starts shifted by 1 nm
    x0, y0 = Config().start_position
    runs = [simulate(Config(mechanism=False, start_position=(x0 + k * 1e-9, y0))) for k in range(5)]
    assert not any(r.fallen for r in runs)
    assert statistics.median(tracking_error_rms(r, 12, 20) for r in runs) < 0.008


def test_mujoco_robot_with_ramps():
    r = simulate(Config())
    assert not r.fallen and r.unreachable_poses == 0
    assert tracking_error_rms(r, 12, 20) < 0.008
    # The fast ramps of the large moves make the ball hop a little: about 3 % of the time, by about 3 mm
    assert r.airborne_fraction < 0.05 and max(r.height) < 0.005


def test_ramps_suppress_bounces():
    # Elastic ball (e ≈ 0.88): without ramps, the jerks of the plate make it bounce; with the gentle ramp
    # only, it never leaves the plate
    assert simulate(Config(max_acceleration=math.inf)).airborne_fraction > 0.5
    assert simulate(Config(fast_move=math.inf)).airborne_fraction < 0.01


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


def test_first_image_after_the_latency():
    # The first measurement arrives 3 periods after the start: until then the plate keeps its command;
    # waiting for the speed, the brain acts at the second image
    cfg = Config(physics="simple", latency=3, duration=0.2)
    acting = lambda r: next(k for k, th in enumerate(r.theta_x) if th != 0.0)
    assert acting(simulate(replace(cfg, wait_for_speed=False))) == 3
    assert acting(simulate(cfg)) == 4


def test_thrown_ball_braked_from_the_first_command():
    # Thrown toward +x: the first command that tilts the plate already brakes the ball. When the first image
    # was given twice (once without delay), the speed seemed zero at the second one and the plate first
    # pushed the ball toward the edge
    r = simulate(Config(trajectory=center, start_position=(-0.1, 0.0), start_velocity=(0.6, 0.0), duration=0.2))
    assert next(th for th in r.theta_x if th != 0.0) < 0


def test_ball_mass_has_no_effect():
    # ẍ = g·sin θ / (1 + k): the mass cancels out
    pm = Config().mujoco_params
    heavy = circle_error(Config(mujoco_params=replace(pm, ball_mass=0.027)))
    assert abs(heavy / circle_error(Config()) - 1) < 0.15


def test_dropped_ball_caught_by_the_aimed_bounce():
    # Dropped from 30.5 cm, 5 cm from the center (MuJoCo, synthetic sensor): caught when the brain aims
    # each bounce; reacting to the ball in the air like on the plate, the brain loses it (with the fast
    # motor ramps, it catches some drops, not this one: the campaign counts them)
    cfg = Config(trajectory=center, start_position=(0.05, 0.0), start_height=0.305, duration=4.0)
    assert not simulate(cfg).fallen
    assert simulate(replace(cfg, airborne_height=math.inf)).fallen


def test_thrown_ball_needs_fast_motors():
    # 0.6 m/s from -10 cm (MuJoCo, synthetic sensor): caught with ramps of 300 rad/s² for the large moves,
    # lost with a single ramp of 50: the plate then needs 0.2 s to flip from the push toward the center to
    # the braking
    cfg = Config(trajectory=center, start_position=(-0.1, 0.0), start_velocity=(0.6, 0.0), duration=5.0)
    assert not simulate(cfg).fallen
    assert simulate(replace(cfg, fast_move=math.inf)).fallen
