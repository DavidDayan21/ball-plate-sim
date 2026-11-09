"""The controller against theory: each derived result is checked.

The closed-loop tests use a single direction of the plate (y = 0) of the simple
physics, whose solution can be computed by hand.
"""
import math
from dataclasses import replace

from controller.pid import PID
from controller.plate import LatencyPredictor
from simulation.config import G, PlateParams
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
