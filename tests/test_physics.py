"""Both physics against the theory of rolling, and the MuJoCo model of the robot."""
from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from controller.kinematics import Geometry, inverse_kinematics
from simulation.config import G, Config, PlateParams
from simulation.mujoco_physics import (MujocoPhysics, rolling_acceleration, build, model_xml, plate_tilt,
                                       measured_restitution)
from simulation.simple_physics import SimplePhysics, acceleration, rotating_plate_acceleration

K = 2 / 3


def test_rolling_term_moves_the_ball_by_minus_r_dtheta():
    # d/dt(ẋ + r·θ̇) = a: tilting the plate by Δθ shifts the ball by exactly -r·Δθ (gravity is the same)
    cfg = Config(physics="simple", plate=PlateParams())
    basic, rolling = SimplePhysics(cfg), SimplePhysics(replace(cfg, rolling_term=True))
    for phys in (basic, rolling):
        phys.command(math.radians(5), 0.0)
        for _ in range(600):
            phys.step(0.0005)
    shift = rolling.ball()[0] - basic.ball()[0]
    assert math.isclose(shift, -cfg.plate.ball_radius * rolling.actual_tilt()[0], rel_tol=1e-6)
    assert rolling.actual_tilt()[0] == pytest.approx(math.radians(5), abs=1e-3)


def test_servo_lag_is_first_order():
    # Motors that reach their target at once: the actual angle follows 1 - exp(-t/τ), τ = kv/kp
    cfg = Config(physics="simple", servo_lag=True, max_speed=1e9, max_acceleration=math.inf)
    tau = cfg.mujoco_params.kv_servo / cfg.mujoco_params.kp_servo
    phys = SimplePhysics(cfg)
    target = inverse_kinematics(cfg.geometry, math.radians(3), 0.0)
    phys.command(math.radians(3), 0.0)
    for _ in range(round(tau / 0.0005)):
        phys.step(0.0005)
    for actual, t in zip(phys.motor_angles(), phys.motors):
        assert math.isclose(actual, t.target * (1 - math.exp(-1)), rel_tol=1e-9)
    assert [m.target for m in phys.motors] == pytest.approx(target, abs=cfg.resolution)


@pytest.mark.parametrize("physics", [SimplePhysics, MujocoPhysics])
def test_angle_driven_plate_follows_ramps(physics):
    # Without arms, the two angles of the plate follow their target with the profile of the motors (here a
    # single ramp of 50 rad/s²), without steps: a jump of the command does not make the plate jump
    cfg = Config(physics="simple", mechanism=False, rolling_term=True, fast_move=math.inf, plate=PlateParams())
    phys = physics(cfg)
    phys.command(0.05, 0.0)
    steps = round(0.02 / cfg.dt_physics)
    for _ in range(steps):
        phys.step(cfg.dt_physics)
    # Still accelerating: at most ½·a·t·(t + dt) (ramp integrated step by step: v then x)
    assert 0.0 < phys.actual_tilt()[0] <= 0.5 * cfg.max_acceleration * 0.02 * (0.02 + cfg.dt_physics) + 1e-12
    for _ in range(10 * steps):
        phys.step(cfg.dt_physics)
    assert phys.actual_tilt() == pytest.approx((0.05, 0.0), abs=1e-4)


def test_rotating_plate_equation_matches_mujoco():
    # Open loop: the plate oscillates (3° at 1 Hz on θx, 2° at 0.7 Hz on θy). From the motion of the
    # plate measured in MuJoCo (Ω, Ω̇, acceleration of O), the full equation predicts the acceleration
    # of the ball; without the r·Ω̇ term, the prediction is far off.
    a, wx, b, wy, dt = math.radians(3), 2 * math.pi, math.radians(2), 1.4 * math.pi, 0.0005
    cg = G / (1 + K)
    phys = MujocoPhysics(Config(plate=PlateParams(), start_position=(0.06, -0.04),
                                start_velocity=(-cg * a / wx, -cg * b / wy)))
    Rs, Os, Ps = [], [], []
    for n in range(round(2.0 / dt)):
        if n % 40 == 0:
            phys.command(a * math.sin(wx * n * dt), b * math.sin(wy * n * dt))
        phys.step(dt)
        Rs.append(phys.data.body("plate").xmat.reshape(3, 3).copy())
        Os.append(phys.data.body("plate").xpos.copy())
        Ps.append(phys.data.body("ball").xpos.copy())
    Rs, Os, Ps = np.array(Rs), np.array(Os), np.array(Ps)
    T = lambda vectors: np.einsum("nji,nj->ni", Rs, vectors)          # world -> plate frame
    p = T(Ps - Os)
    W = np.einsum("nji,njk->nik", Rs, np.gradient(Rs, dt, axis=0))  # Rᵀ·Ṙ = [Ω]×
    omega = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    domega = np.gradient(omega, dt, axis=0)
    v = np.gradient(p, dt, axis=0)
    acc = np.gradient(v, dt, axis=0)
    gravity = T(np.tile([0.0, 0.0, -G], (len(p), 1))) - T(np.gradient(np.gradient(Os, dt, axis=0), dt, axis=0))
    r = 0.02
    measured = np.stack([acc[:, 0] + r * domega[:, 1], acc[:, 1] - r * domega[:, 0]], axis=1)
    predicted = np.array([rotating_plate_acceleration((x, y, r), (vx, vy), o, do, g[:2], K)
                          for (x, y, _), (vx, vy, _), o, do, g in zip(p, v, omega, domega, gravity)])
    smooth = lambda s: np.array([np.convolve(c, np.ones(40) / 40, mode="same") for c in s.T]).T[400:-40]
    rms = lambda s: math.sqrt(np.mean(np.sum(smooth(s) ** 2, axis=1)))
    assert rms(measured - predicted) < 0.02 * rms(measured)
    without_rolling = acc[:, :2]
    assert rms(without_rolling - predicted) > 0.2 * rms(measured)


def test_simple_physics_hollow_ball():
    # ẍ = g·sin θ / (1 + k) = (3/5)·g·sin θ for k = 2/3, independent axes
    ax, ay = acceleration(math.radians(2), 0.0, PlateParams())
    assert math.isclose(ax, 0.6 * G * math.sin(math.radians(2)), rel_tol=1e-12)
    assert ay == 0.0


@pytest.mark.parametrize("k", [2 / 3, 2 / 5])
def test_mujoco_matches_rolling_law(k):
    theta = math.radians(2)
    assert abs(rolling_acceleration(k, theta) / (G * math.sin(theta) / (1 + k)) - 1) < 0.01


def test_no_slip_condition():
    # The ball rolls if μ ≥ k·tan θ / (1 + k), otherwise it slides: a = g·(sin θ - μ·cos θ)
    th = math.radians(10)
    mu_min = K * math.tan(th) / (1 + K)
    rolls = rolling_acceleration(K, th, friction=1.3 * mu_min)
    slides = rolling_acceleration(K, th, friction=0.7 * mu_min)
    assert abs(rolls / (G * math.sin(th) / (1 + K)) - 1) < 0.01
    assert abs(slides / (G * (math.sin(th) - 0.7 * mu_min * math.cos(th))) - 1) < 0.02


def _ball_on_fixed_plate(theta, friction, duration=0.3, vx=0.0, rx_spin=None, k_actual=None):
    """Simple physics, sliding allowed: angle-driven plate held at θx (no bias), ball released at the center
    with the speed vx. Returns (times, x, phys)."""
    pm = replace(Config().mujoco_params, friction=friction, k_actual=k_actual)
    cfg = Config(physics="simple", mechanism=False, sliding=True, plate=PlateParams(), mujoco_params=pm,
                 start_position=(0.0, 0.0), start_velocity=(vx, 0.0))
    phys = SimplePhysics(cfg)
    phys.tilt = phys.previous_tilt = (theta, 0.0)
    for m in phys.motors:
        m.position = m.target = theta if m is phys.motors[0] else 0.0
    if rx_spin is not None:
        phys.spin = (0.0, rx_spin)                 # r·ωy given: the ball starts with this spin, sliding
    T, X = [], []
    for n in range(round(duration / 0.0005)):
        phys.step(0.0005)
        T.append((n + 1) * 0.0005)
        X.append(phys.ball()[0])
    return np.array(T), np.array(X), phys


def test_simple_physics_sliding_matches_the_no_slip_condition():
    # Same theory as MuJoCo above: a = g·sin θ / (1 + k) above μ_min, a = g·(sin θ - μ·cos θ) below
    th = math.radians(10)
    mu_min = K * math.tan(th) / (1 + K)
    for factor, expected in ((1.3, G * math.sin(th) / (1 + K)), (0.7, G * (math.sin(th) - 0.7 * mu_min * math.cos(th)))):
        t, x, phys = _ball_on_fixed_plate(th, factor * mu_min)
        assert 2 * np.polyfit(t, x, 2)[0] == pytest.approx(expected, rel=1e-3)
        assert (phys.spin is None) == (factor > 1)
    # A ball inside k_actual: solid ball, c = 5/7
    t, x, _ = _ball_on_fixed_plate(th, 1.0, k_actual=2 / 5)
    assert 2 * np.polyfit(t, x, 2)[0] == pytest.approx(G * math.sin(th) / (1 + 2 / 5), rel=1e-3)


def test_sliding_ball_rolls_again():
    # Level plate, ball thrown at 0.5 m/s without spin: friction slows it and spins it up until it rolls,
    # at v = v0/(1 + k) (angular momentum about the contact point conserved), after t = k·v0 / ((1 + k)·μ·g)
    v0, mu = 0.5, 0.05
    t, x, phys = _ball_on_fixed_plate(0.0, mu, duration=0.6, vx=v0, rx_spin=0.0)
    assert phys.spin is None
    assert phys.ball_velocity()[0] == pytest.approx(v0 / (1 + K), rel=1e-3)
    t_roll = K * v0 / ((1 + K) * mu * G)
    x_roll = v0 * t_roll - mu * G * t_roll ** 2 / 2
    assert phys.ball()[0] == pytest.approx(x_roll + v0 / (1 + K) * (0.6 - t_roll), rel=1e-3)


def test_friction_cone_option():
    # Elliptic (Coulomb's round cone) by default; pyramidal: MuJoCo's own default, no attribute in the model
    assert mujoco.MjModel.from_xml_string(model_xml(Config())).opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    pyramid = Config(mujoco_params=replace(Config().mujoco_params, cone="pyramidal"))
    assert "cone=" not in model_xml(pyramid)
    assert mujoco.MjModel.from_xml_string(model_xml(pyramid)).opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


def test_dropped_ball():
    with pytest.raises(ValueError):                       # the simple physics keeps the ball on the plate
        SimplePhysics(Config(physics="simple", start_height=0.1))
    phys = MujocoPhysics(Config(start_height=0.305, start_position=(0.05, 0.0)))
    assert phys.ball_height() == pytest.approx(0.305, abs=1e-9) and phys.ball() == pytest.approx((0.05, 0.0))


def test_fall_detection():
    phys = MujocoPhysics(Config())
    R, r = phys.p.plate_radius, phys.p.ball_radius
    phys.place_ball(R + 0.01, 0.0, 0.05)        # beyond the edge but 5 cm above the plate: may still come back
    assert not phys.fallen()
    phys.place_ball(R + 0.01, 0.0, 0.0)         # beyond the edge, level with the plate
    assert phys.fallen()
    phys.place_ball(0.0, 0.0, -r - 0.001)       # under the surface
    assert phys.fallen()


def test_bounce_complies_with_ittf_rule():
    # Dropped from 30.5 cm, a regulation ping-pong ball bounces back to 22-25 cm: e ∈ [0.85, 0.91]
    assert 0.85 <= measured_restitution(height=0.305) <= 0.91


@pytest.mark.parametrize("mechanism", [True, False])
def test_mujoco_start_like_a_robot_already_on(mechanism):
    # The plate already holds the command 0 (its calibration error) and is at rest; the ball is placed at its
    # start position and velocity in the plate frame, on the surface; the time starts at 0
    cfg = Config(mechanism=mechanism, start_position=(0.05, -0.02), start_velocity=(0.3, 0.1))
    phys = MujocoPhysics(cfg)
    tx, ty = phys.actual_tilt()                       # tilt minus the calibration error
    assert abs(math.degrees(tx)) < 0.02 and abs(math.degrees(ty)) < 0.02
    assert abs(math.degrees(plate_tilt(phys.data)[0]) - 0.5) < 0.02     # the calibration error of θx
    assert np.linalg.norm(phys.data.body("plate").cvel[:3]) < 1e-3        # at rest (rad/s)
    assert phys.ball() == pytest.approx((0.05, -0.02), abs=1e-12)
    assert phys.ball_velocity() == pytest.approx((0.3, 0.1), abs=1e-12)
    assert abs(phys.ball_height()) < 1e-12 and phys.data.time == 0.0 and phys.airborne_steps == 0


def test_mujoco_robot_at_rest_plate_level():
    model, data = build(Config())
    for _ in range(1000):
        mujoco.mj_step(model, data)
    tx, ty = plate_tilt(data)
    assert abs(math.degrees(tx)) < 0.01 and abs(math.degrees(ty)) < 0.01
    assert abs(data.body("plate").xpos[2] - Geometry().height) < 0.001


def test_inverse_kinematics_validated_by_mujoco():
    # MuJoCo's closed chain (with its parasitic motions) must give the desired tilt
    model, data = build(Config())
    data.ctrl[:] = inverse_kinematics(Geometry(), math.radians(5), math.radians(-3))
    for _ in range(2000):
        mujoco.mj_step(model, data)
    tx, ty = plate_tilt(data)
    assert abs(math.degrees(tx) - 5) < 0.05 and abs(math.degrees(ty) + 3) < 0.05
