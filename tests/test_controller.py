"""The controller against theory: each derived result is checked.

The closed-loop tests use a single direction of the plate (y = 0) of the simple
physics, whose solution can be computed by hand.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from controller.adaptation import ModelGainEstimator
from controller.flight import FlightEstimator
from controller.lqr import lqr, pid_gains
from controller.pid import PID
from controller.plate import LatencyPredictor
from controller.network import Network
from controller.policy import LAYOUT, OBSERVATION_SIZE, PolicyController
from controller.ramps import FastRamps
from simulation.camera import inverse_rotation
from simulation.config import G, REFERENCE_GAINS, Config, PlateParams, circle
from simulation.loop import pid_brain
from simulation.simple_physics import acceleration

P = PlateParams()
CG = P.coefficient * G


def loop_1d(pid, p=P, x0=0.10, dt=0.01, duration=10.0):
    """Ball on a single direction of the plate, semi-implicit Euler. Returns (times, positions, fallen)."""
    x, v = x0, 0.0
    T, X = [], []
    for n in range(round(duration / dt) + 1):
        theta = pid.compute(0.0, x, dt)
        T.append(n * dt)
        X.append(x)
        if abs(x) > p.plate_radius:
            return T, X, True
        v += acceleration(theta, 0.0, p)[0] * dt
        x += v * dt
    return T, X, False


def kd_for(zeta, kp=1.0):
    return 2 * zeta * math.sqrt(CG * kp) / CG


def test_pid_basic_computation():
    pid = PID(kp=2.0, ki=0.5, kd=0.3)
    u1 = pid.compute(1.0, 0.0, 0.1)          # e = 1, derivative = 0, integral = 0.1
    u2 = pid.compute(1.0, 0.5, 0.1)          # e = 0.5, derivative = -5, integral = 0.15
    assert math.isclose(u1, 2.0 + 0.05, rel_tol=1e-12)
    assert math.isclose(u2, 1.0 + 0.075 - 1.5, rel_tol=1e-12)


def test_anti_windup_freezes_integral():
    pid = PID(kp=1.0, ki=1.0, output_min=-0.1, output_max=0.1)
    for _ in range(100):
        assert pid.compute(1.0, 0.0, 0.01) == 0.1
    assert pid.integral == 0.0


def test_p_controller_period():
    # Harmonic oscillator: T0 = 2π / √(c·g·Kp) (small amplitude: linear range)
    kp = 0.5
    T, X, fallen = loop_1d(PID(kp=kp), x0=0.01)
    crossings = [T[i] for i in range(1, len(X)) if X[i - 1] * X[i] < 0]
    period = 2 * (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    assert not fallen
    assert math.isclose(period, 2 * math.pi / math.sqrt(CG * kp), rel_tol=0.01)


def test_semi_implicit_euler_preserves_amplitude():
    _, X, _ = loop_1d(PID(kp=1.0))
    assert max(abs(x) for x in X[-300:]) < 0.1005


def test_pd_overshoot():
    # D = exp(-π·ζ / √(1 - ζ²))
    zeta = 0.7
    _, X, _ = loop_1d(PID(kp=1.0, kd=kd_for(zeta)))
    assert abs(max(0.0, -min(X)) / 0.10 - math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))) < 0.005


def test_pd_steady_state_error_and_pid_removes_it():
    bias = math.radians(0.5)
    p = replace(P, bias_x=bias)
    _, X_pd, _ = loop_1d(PID(kp=1.0, kd=kd_for(0.7)), p=p, duration=20)
    _, X_pid, _ = loop_1d(PID(kp=1.0, ki=1.0, kd=kd_for(0.7)), p=p, duration=20)
    assert math.isclose(X_pd[-1], bias / 1.0, rel_tol=1e-3)      # e = θb / Kp
    assert abs(X_pid[-1]) < 1e-4


def test_integration_zone():
    pid = PID(kp=1.0, ki=1.0, integration_zone=0.02)
    pid.compute(0.0, 0.05, 0.1)                 # |e| = 5 cm, outside the zone: nothing is integrated
    assert pid.integral == 0.0
    pid.compute(0.0, 0.01, 0.1)                 # |e| = 1 cm, inside the zone
    assert math.isclose(pid.integral, -0.001, rel_tol=1e-12)


def test_integral_everywhere_must_be_paid_back_by_an_overshoot():
    # At equilibrium, Ki·∫e = -θb (the integral cancels the calibration error), so ∫x·dt = θb/Ki whatever
    # the transient: the area built up during the approach from 10 cm must be paid back by an overshoot
    bias = math.radians(0.5)
    T, X, _ = loop_1d(PID(kp=1.0, ki=1.0, kd=kd_for(0.7)), p=replace(P, bias_x=bias), duration=20)
    dt = T[1] - T[0]
    positive, negative = sum(x for x in X if x > 0) * dt, sum(x for x in X if x < 0) * dt
    assert math.isclose(positive + negative, bias / 1.0, rel_tol=1e-4)
    assert -negative > 0.5 * positive and -min(X) > 0.040


def test_integration_zone_suppresses_the_overshoot_and_still_removes_the_bias():
    # The integral only starts near the setpoint, where it still cancels the offset θb/Kp of the
    # calibration error (8.7 mm for 0.5°, inside the 2 cm zone)
    p = replace(P, bias_x=math.radians(0.5))
    _, X, _ = loop_1d(PID(kp=1.0, ki=1.0, kd=kd_for(0.7), integration_zone=0.02), p=p, duration=20)
    assert -min(X) < 0.001 and abs(X[-1]) < 1e-4


def test_integration_zone_cannot_correct_a_larger_calibration_error():
    # Offset θb/Kp = 26 mm > zone: the ball stops outside the zone, the integral never starts
    bias = math.radians(1.5)
    _, X, _ = loop_1d(PID(kp=1.0, ki=1.0, kd=kd_for(0.7), integration_zone=0.02), p=replace(P, bias_x=bias),
                      duration=20)
    assert math.isclose(X[-1], bias / 1.0, rel_tol=1e-3)


def test_fast_ramps_only_for_a_fast_ball():
    # The fast ramps of the motors are allowed only for a ball faster than fast_speed (speed from the last two
    # measurements): never before the first speed is known
    assert not FastRamps(0.3).allowed
    for speed, allowed in ((0.1, False), (0.5, True)):
        ramps = FastRamps(0.3)
        assert not ramps.update((-0.1, 0.0), 0.02)
        for k in (1, 2):
            assert ramps.update((-0.1 + speed * 0.02 * k, 0.0), 0.02) == ramps.allowed == allowed
    assert FastRamps(0.0).update((0.0, 0.0), 0.02)             # fast_speed = 0: whatever the speed


def test_lqr_double_integrator():
    # ẍ = u with the cost ∫(q·x² + u²)·dt: known solution K = (√q, √(2·√q))
    q = 4.0
    K = lqr(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), np.diag([q, 0.0]), np.eye(1))
    assert np.allclose(K, [[math.sqrt(q), math.sqrt(2 * math.sqrt(q))]], rtol=1e-12, atol=0)


def test_lqr_gains_of_the_pid():
    # Closed loop of one axis: s³ + c·g·Kd·s² + c·g·Kp·s + c·g·Ki = 0, stable
    kp, ki, kd = pid_gains(CG, 0.14, 0.7, math.radians(10))
    assert np.all(np.roots([1, CG * kd, CG * kp, CG * ki]).real < 0)
    # When the integral is slow compared with the loop, its pole is -1/integral_time
    kp, ki, kd = pid_gains(CG, 0.05, 5.0, math.radians(10))
    slowest = max(np.roots([1, CG * kd, CG * kp, CG * ki]).real)
    assert math.isclose(slowest, -1 / 5.0, rel_tol=1e-4)


def test_gains_of_the_configuration():
    cfg = Config()
    assert cfg.pid_gains() == pid_gains(CG, cfg.lqr_position, cfg.lqr_integral_time, cfg.lqr_tilt)
    assert cfg.pid_gains() == pytest.approx((1.89, 1.75, 0.80), abs=0.005)      # chosen in B3
    assert Config(gains=(1.0, 2.0, 3.0)).pid_gains() == (1.0, 2.0, 3.0)
    # Hand-tuned gains of the frozen reference controller: Kd gives the damping ζ = 0.7 to the PD
    assert REFERENCE_GAINS == (1.0, 1.0, kd_for(0.7))


def test_pid_unstable_beyond_routh():
    # Stable if and only if Ki < c·g·Kp·Kd
    kd = kd_for(0.7)
    limit = CG * 1.0 * kd
    _, _, fell_stable = loop_1d(PID(kp=1.0, ki=0.8 * limit, kd=kd), x0=0.05, duration=30)
    _, _, fell_unstable = loop_1d(PID(kp=1.0, ki=1.2 * limit, kd=kd), x0=0.05, duration=30)
    assert not fell_stable
    assert fell_unstable


def test_predictor_exact_if_model_exact():
    T, g, N = 0.02, 5.886, 3
    p = LatencyPredictor(N, T, g)
    x, v, history = 0.05, 0.1, []
    for k in range(300):
        history.append(x)
        estimate = p.predict(history[max(0, k - N)])
        u = 0.05 * math.sin(0.3 * k)
        p.store(u)
        v += g * u * T
        x += v * T
    assert abs(estimate - history[-1]) < 1e-9



# --- Ball in the air: ballistic estimator (controller/flight.py) ------------------------------------
PLATE_H, BALL_R, T_CTRL = 0.12, 0.02, 0.02
LEVEL = PLATE_H + BALL_R                  # height of the center of a ball resting on the level plate


def flight_images(p0, v0, theta, times, t_start=0.0, sigma=(0.0, 0.0, 0.0), seed=0):
    """Plate-frame measurements (x, y, height) at the given times of a ball in free flight, at p0 with the
    velocity v0 (world frame) at t_start, the plate held at the tilt theta."""
    rng = np.random.default_rng(seed)
    images = []
    for t in times:
        tau = t - t_start
        p = np.array(p0) + np.array(v0) * tau + np.array([0.0, 0.0, -G * tau * tau / 2 - PLATE_H])
        x, y, z = inverse_rotation(*theta, p)
        images.append(tuple(np.array([x, y, z - BALL_R]) + rng.normal(0.0, sigma, 3)))
    return images


def level_impact(p0, v0):
    """Time, position and velocity at which the free parabola reaches the level of a level plate."""
    t = (v0[2] + math.sqrt(v0[2] ** 2 + 2 * G * (p0[2] - LEVEL))) / G
    return t, (p0[0] + v0[0] * t, p0[1] + v0[1] * t), (v0[0], v0[1], v0[2] - G * t)


def track(images, theta, latency):
    """Estimator after one observation per image (image j delivered at the period j + latency, like the
    sensor), the plate held at the tilt theta since before the start."""
    est = FlightEstimator(PLATE_H, BALL_R, T_CTRL, latency, airborne_height=0.015, g=G)
    for _ in range(latency + 1):
        est.command(theta)
    for image in images:
        est.observe(image)
        est.command(theta)
    return est


def test_flight_exact_parabola():
    # Tilted plate, measurements 2 periods late: the fit finds the parabola again exactly
    p0, v0, theta = (0.05, -0.03, 0.40), (0.10, 0.05, 0.5), (math.radians(5), math.radians(-3))
    est = track(flight_images(p0, v0, theta, [j * T_CTRL for j in range(8)]), theta, latency=2)
    t, (x, y), v = level_impact(p0, v0)
    dt, p, vi = est.impact()
    assert dt == pytest.approx(t - 9 * T_CTRL, abs=1e-9) and p[:2] == pytest.approx((x, y), abs=1e-9)
    assert vi == pytest.approx(v, abs=1e-9)


def test_flight_with_noise():
    # Drop from 30 cm, noise like the camera (1 mm on x and y, 3 mm on the height), 6 images
    p0, v0, theta = (0.05, -0.03, LEVEL + 0.30), (0.0, 0.0, 0.0), (0.0, 0.0)
    images = flight_images(p0, v0, theta, [j * T_CTRL for j in range(6)], sigma=(0.001, 0.001, 0.003), seed=4)
    dt, p, vi = track(images, theta, latency=1).impact()
    t, (x, y), v = level_impact(p0, v0)
    assert abs(dt - (t - 6 * T_CTRL)) < 0.01 and math.hypot(p[0] - x, p[1] - y) < 0.005
    assert vi[2] == pytest.approx(v[2], rel=0.05)


def test_flight_repeated_measurement_ignored():
    # The camera repeats its last measurement when it does not see the ball (images 4 and 5 here):
    # these are not points of the flight, and the next images keep their true times
    p0, v0 = (0.03, 0.0, 0.40), (0.05, 0.0, 0.0)
    images = flight_images(p0, v0, (0.0, 0.0), [j * T_CTRL for j in range(8)])
    est = track(images[:4] + [images[3]] * 2 + images[6:], (0.0, 0.0), latency=1)
    assert len(est.samples) == 6
    t, (x, y), _ = level_impact(p0, v0)
    assert est.impact()[0] == pytest.approx(t - 8 * T_CTRL, abs=1e-9) and est.impact()[1][0] == pytest.approx(x)


def test_flight_on_the_plate_and_bounce():
    est = FlightEstimator(PLATE_H, BALL_R, T_CTRL, 0, airborne_height=0.015, g=G)
    est.observe((0.01, 0.0, 0.002))
    assert est.impact() is None                               # ball on the plate
    # Drop of 16 cm, bounce at 1.6 m/s; no image near the plate (the one at 0.18 s is missed: repeated)
    t1, _, _ = level_impact((0.0, 0.0, LEVEL + 0.16), (0.0, 0.0, 0.0))
    first = flight_images((0.0, 0.0, LEVEL + 0.16), (0.0, 0.0, 0.0), (0.0, 0.0), [j * T_CTRL for j in range(9)])
    second = flight_images((0.0, 0.0, LEVEL), (0.0, 0.0, 1.6), (0.0, 0.0), [j * T_CTRL for j in range(10, 14)],
                           t_start=t1)
    est = track(first + [first[-1]] + second, (0.0, 0.0), latency=0)
    assert est.impact()[0] == pytest.approx(t1 + 2 * 1.6 / G - 13 * T_CTRL, abs=1e-9)


def test_pid_hold_follows_without_acting():
    pid = PID(kp=2.0, ki=1.0, kd=0.5)
    pid.compute(0.0, 0.01, 0.02)
    integral = pid.integral
    u = pid.compute(0.0, 0.03, 0.02, hold=True)             # in the air: nothing integrated, integral term only
    assert pid.integral == integral and u == pytest.approx(1.0 * integral)
    assert pid.previous_error == -0.03 and pid.derivative == pytest.approx(-1.0)   # (-0.03 + 0.01) / 0.02


def test_aimed_bounce():
    # Ball dropped 5 cm from the center: level plate as long as the impact is not estimated, then the tilt
    # that sends it back halfway to the center: θ = (0.5·(0 - x) / (2·e·v_i/g) - 0) / ((1 + e)·v_i)
    from controller.plate import PlateController
    e = 0.87
    flight = FlightEstimator(PLATE_H, BALL_R, T_CTRL, 0, airborne_height=0.015, g=G)
    ctrl = PlateController(1.0, 1.0, 0.5, math.radians(10), 0.05, CG, flight=flight, restitution=e, bounce_gain=0.5)
    p0, v0 = (0.05, 0.0, LEVEL + 0.20), (0.0, 0.0, 0.0)
    images = flight_images(p0, v0, (0.0, 0.0), [0.0, T_CTRL])
    assert ctrl.compute((0.0, 0.0), images[0], (0.0, 0.0), T_CTRL) == (0.0, 0.0)
    theta = ctrl.compute((0.0, 0.0), images[1], (0.0, 0.0), T_CTRL)
    v_i = math.sqrt(2 * G * 0.20)
    expected = 0.5 * (0.0 - 0.05) / (2 * e * v_i / G) / ((1 + e) * v_i)
    assert theta[0] == pytest.approx(expected, rel=1e-6) and theta[0] < 0 and theta[1] == pytest.approx(0.0, abs=1e-9)


def test_pid_waits_for_the_speed():
    # First measurement: feedforward and integral term only; second: the derivative starts at the raw speed
    pid = PID(kp=2.0, ki=1.0, kd=0.5, tau_d=0.05, wait_for_speed=True)
    assert pid.compute(0.0, 0.10, 0.02, feedforward=0.01) == 0.01
    pid.compute(0.0, 0.11, 0.02)
    assert pid.derivative == pytest.approx(-0.5)                # (-0.11 + 0.10) / 0.02, no lag of the filter
    plain = PID(kp=2.0, ki=1.0, kd=0.5, tau_d=0.05)
    plain.compute(0.0, 0.10, 0.02)
    plain.compute(0.0, 0.11, 0.02)
    assert plain.derivative == pytest.approx(-0.5 * 0.02 / 0.07)   # filtered: (1 - a)·raw, a = 0.05 / 0.07


# --- Adaptive control: estimation of the model gain c·g (B6) ------------------------------------
def circle_with_estimator(true_gain, nominal_gain=CG, latency=1, duration=12.0, offset=(0.0, 0.0),
                          airborne_every=0, bias=(0.0087, -0.0052)):
    """Ball on the circle of the reference scenario, model of the estimator exact: on each axis
    x(j+1) - 2·x(j) + x(j-1) = T²·G·((u(j-1) + u(j))/2 + θb). PD on the true position error, feedforward with the
    nominal gain; the estimator receives the positions N periods late. Returns the estimator."""
    T = 0.02
    previous_setpoint = np.array(circle(-T)[:2])
    est = ModelGainEstimator(nominal_gain, T, latency, math.radians(10))
    x = [np.array([0.06 * math.cos(math.pi / 4), 0.06 * math.sin(math.pi / 4)])] * 2   # x(j-1), x(j)
    u_prev = np.zeros(2)
    history = [x[1]] * (latency + 1)
    for j in range(round(duration / T)):
        xr, yr, axr, ayr = circle(j * T)
        setpoint = np.array([xr, yr])
        est.update(tuple(history[-(latency + 1)]), tuple(setpoint + offset),
                   airborne=airborne_every > 0 and j % airborne_every == 0)
        error_speed = (setpoint - previous_setpoint - x[1] + x[0]) / T
        previous_setpoint = setpoint
        u = np.array([axr, ayr]) / nominal_gain + 1.89 * (setpoint - x[1]) + 0.8 * error_speed
        est.command(tuple(u))
        x = [x[1], 2 * x[1] - x[0] + T ** 2 * true_gain * ((u_prev + u) / 2 + np.array(bias))]
        u_prev = u
        history.append(x[1])
    return est


def test_estimator_finds_the_gain_of_the_ball():
    solid = G / (1 + 2 / 5)                          # c·g of a solid ball, 19 % above the nominal (hollow) one
    for latency in (1, 3):
        assert circle_with_estimator(solid, latency=latency).gain == pytest.approx(solid, rel=0.005)
    assert circle_with_estimator(CG).gain == pytest.approx(CG, rel=0.005)


def test_estimator_waits_for_a_calm_rolling_ball():
    solid = G / (1 + 2 / 5)
    # Ball 2 cm away from its setpoint, or in the air every 0.5 s: the gain keeps its nominal value
    assert circle_with_estimator(solid, offset=(0.02, 0.0)).gain == CG
    assert circle_with_estimator(solid, airborne_every=25).gain == CG
    # Ball at rest on its setpoint: nothing to learn either
    est = ModelGainEstimator(CG, 0.02, 1, math.radians(10))
    for _ in range(500):
        est.update((0.01, 0.0), (0.01, 0.0), airborne=False)
        est.command((0.0, 0.0))
    assert est.gain == CG


# --- Brain driven by a policy (controller/policy.py) ----------------------------------------------
def test_policy_observation():
    T, theta_max = 0.02, math.radians(10)
    brain = PolicyController(None, theta_max, latency=3)
    # First measurement: no speed yet, no command sent
    o = brain.observe((0.05, 0.0), (0.02, -0.01, 0.004), (0.3, 0.0), T)
    assert o.shape == (OBSERVATION_SIZE,) and o.dtype == np.float32
    assert o[LAYOUT["position"]] == pytest.approx([0.2, -0.1])           # in units of 10 cm
    assert o[LAYOUT["heights"]] == pytest.approx([0.0] * 8)              # 4 mm: on the plate
    assert o[LAYOUT["setpoint"]] == pytest.approx([0.5, 0.0])
    assert o[LAYOUT["setpoint_acceleration"]] == pytest.approx([0.3, 0.0])
    assert o[LAYOUT["latency"]] == pytest.approx(3 / 5) and o[LAYOUT["received"]] == pytest.approx(1 / 8)
    assert not (o[LAYOUT["speeds"]].any() or o[LAYOUT["commands"]].any() or o[LAYOUT["setpoint_speed"]].any())
    # The action is the command in units of theta_max, saturated at ±1
    assert brain.act((0.5, -3.0)) == pytest.approx((0.5 * theta_max, -theta_max))
    # Second measurement: the ball has moved by 2 mm (0.1 m/s), the setpoint by 1 mm (0.05 m/s)
    o = brain.observe((0.051, 0.0), (0.022, -0.01, 0.0), (0.0, 0.0), T)
    assert o[LAYOUT["speeds"]] == pytest.approx([0.5, 0.0] + [0.0] * 12)  # in units of 0.2 m/s; 7 speeds (x, y)
    assert o[LAYOUT["setpoint_speed"]] == pytest.approx([0.25, 0.0])
    assert o[LAYOUT["commands"]] == pytest.approx([0.5, -1.0] + [0.0] * 22)        # the last 12 commands
    assert o[LAYOUT["heights"]] == pytest.approx([0.0] * 8)
    assert o[LAYOUT["received"]] == pytest.approx(2 / 8)
    # Third measurement: speeds and commands newest first; a jump of the setpoint is bounded
    brain.act((0.1, 0.2))
    o = brain.observe((-0.05, 0.0), (0.022, -0.012, 0.0), (0.0, 0.0), T)
    assert o[LAYOUT["speeds"]][:6] == pytest.approx([0.0, -0.5, 0.5, 0.0, 0.0, 0.0], abs=1e-6)
    assert o[LAYOUT["commands"]][:4] == pytest.approx([0.1, 0.2, 0.5, -1.0])
    assert o[LAYOUT["setpoint_speed"]] == pytest.approx([-2.5, 0.0])
    assert o[LAYOUT["received"]] == pytest.approx(3 / 8)
    # The error compares the ball with the setpoint of the time of its measurement: here, still the first one
    # (latency of 3 periods), 5 cm - 2.2 cm, bounded at 2 cm
    assert o[LAYOUT["error"]] == pytest.approx([2.0, 1.2])
    for _ in range(5):
        o = brain.observe((-0.05, 0.0), (-0.045, 0.0, 0.0), (0.0, 0.0), T)
    assert o[LAYOUT["error"]] == pytest.approx([-0.5, 0.0]) and o[LAYOUT["received"]] == 1.0


def test_policy_observation_of_a_ball_in_the_air():
    # Heights in units of 2 cm, newest first; 0 up to 15 mm: on the plate, only the error of the sensor
    brain = PolicyController(None, 0.1, latency=1)
    for height in (0.30, 0.20, 0.016, 0.014, -0.002):
        o = brain.observe((0.0, 0.0), (0.0, 0.0, height), (0.0, 0.0), 0.02)
    assert o[LAYOUT["heights"]] == pytest.approx([0.0, 0.0, 0.8, 10.0, 15.0, 15.0, 15.0, 15.0])
    # Another threshold, as given by the configuration (infinite: never in the air)
    brain = PolicyController(None, 0.1, latency=1, airborne_height=math.inf)
    assert not brain.observe((0.0, 0.0), (0.0, 0.0, 0.30), (0.0, 0.0), 0.02)[LAYOUT["heights"]].any()


def test_policy_observation_of_the_ball():
    # The estimate of c·g, as its relative gap to the value assumed at the start, in units of 0.2
    class Estimator:
        gain, seen, sent = 6.0, [], []

        def update(self, measured, setpoint, airborne):
            self.seen.append((measured, setpoint, airborne))

        def command(self, theta):
            self.sent.append(theta)

    estimator = Estimator()
    brain = PolicyController(lambda observation: (0.5, -2.0), 0.1, latency=1, estimator=estimator)
    o = brain.observe((0.01, 0.0), (0.02, 0.03, 0.0), (0.0, 0.0), 0.02)
    assert o[LAYOUT["ball"]] == pytest.approx([0.0]) and brain.model_gain == 6.0
    estimator.gain = 7.2                                # the estimator has learned a ball 20 % more lively
    o = brain.observe((0.01, 0.0), (0.02, 0.03, 0.1), (0.0, 0.0), 0.02)
    assert o[LAYOUT["ball"]] == pytest.approx([1.0]) and brain.model_gain == 7.2
    # The estimator is given every measurement (in the air or not) and every command really sent
    assert estimator.seen == [((0.02, 0.03), (0.01, 0.0), False), ((0.02, 0.03), (0.01, 0.0), True)]
    brain.compute((0.01, 0.0), (0.02, 0.03, 0.0), (0.0, 0.0), 0.02)
    assert estimator.sent == [(pytest.approx(0.05), pytest.approx(-0.1))]
    # Without estimator: the ball assumed
    brain = PolicyController(None, 0.1, latency=1)
    assert brain.observe((0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0), 0.02)[LAYOUT["ball"]] == 0.0


def test_policy_observation_error_and_integral():
    T = 0.02
    brain = PolicyController(None, 0.1, latency=1)
    # x: 5 mm from its setpoint during 1 s; y: 3 cm away, outside the integration zone
    for _ in range(50):
        o = brain.observe((0.005, 0.03), (0.0, 0.0), (0.0, 0.0), T)
    assert o[LAYOUT["error"]] == pytest.approx([0.5, 2.0])               # in units of 1 cm, bounded at ±2
    assert o[LAYOUT["integral"]] == pytest.approx([0.5, 0.0])            # in units of 1 cm·s
    # The integral is bounded at ±3
    for _ in range(400):
        o = brain.observe((0.0, 0.0), (0.019, 0.0), (0.0, 0.0), T)
    assert o[LAYOUT["error"]] == pytest.approx([-1.9, 0.0]) and o[LAYOUT["integral"]] == pytest.approx([-3.0, 0.0])


def test_pid_brain_told_the_command_really_applied():
    # Imitation: a pupil drives, the PID brain only advises; its memories of the commands hold what was sent
    brain = pid_brain(Config(latency=2))
    brain.compute((0.0, 0.0), (0.03, 0.01, 0.0), (0.0, 0.0), 0.02)
    advice = brain.compute((0.0, 0.0), (0.03, 0.01, 0.0), (0.0, 0.0), 0.02)
    assert brain.predictors[0].commands[-1] == advice[0] != 0.0
    brain.applied((0.01, -0.02))
    assert [predictor.commands[-1] for predictor in brain.predictors] == [0.01, -0.02]
    assert brain.flight.commands[-1] == brain.estimator.commands[-1] == (0.01, -0.02)
    assert len(brain.predictors[0].commands) == 3 and brain.predictors[0].commands[-2] == 0.0   # older ones kept


def test_network(tmp_path):
    rng = np.random.default_rng(0)
    layers = [(rng.normal(size=(4, 3)), rng.normal(size=4)), (rng.normal(size=(2, 4)), rng.normal(size=2))]
    network, x = Network(layers), np.array([0.2, -1.0, 0.5], dtype=np.float32)
    assert network(x) == pytest.approx(layers[1][0] @ np.tanh(layers[0][0] @ x + layers[0][1]) + layers[1][1])
    network.save(tmp_path / "policy.npz")                               # same function once saved and read
    assert np.array_equal(Network.load(tmp_path / "policy.npz")(x), network(x))


def test_policy_brain_has_the_interface_of_the_pid_brain():
    seen = []

    def policy(observation):
        seen.append(observation)
        return 0.2, -0.4

    brain = PolicyController(policy, 0.1, latency=1)
    assert brain.compute((0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.0), 0.02) == pytest.approx((0.02, -0.04))
    assert brain.compute((0.0, 0.0), (0.01, 0.0), (0.0, 0.0), 0.02) == pytest.approx((0.02, -0.04))
    assert seen[1][LAYOUT["commands"]][:2] == pytest.approx([0.2, -0.4]) and math.isnan(brain.model_gain)
    with pytest.raises(ValueError):
        PolicyController(policy, 0.1, latency=6)               # the policy knows up to 5 periods of latency
    with pytest.raises(ValueError):
        brain.act((math.nan, 0.0))
