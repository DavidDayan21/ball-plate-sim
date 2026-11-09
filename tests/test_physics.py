"""Both physics against the theory of rolling, and the MuJoCo model of the robot."""
from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from controller.kinematics import Geometry, inverse_kinematics
from simulation.config import G, Config, PlateParams
from simulation.mujoco_physics import MujocoPhysics, rolling_acceleration, build, plate_tilt, measured_restitution
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


def test_additions_need_the_mechanism():
    with pytest.raises(ValueError):
        SimplePhysics(Config(physics="simple", mechanism=False, rolling_term=True))


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
