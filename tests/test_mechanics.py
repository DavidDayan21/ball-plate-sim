"""Kinematics of the 3-arm mechanism and motor model."""
import math

import pytest

from controller.kinematics import Geometry, forward_kinematics, inverse_kinematics
from simulation.motor import Motor

G3 = Geometry()


def test_rest_arms_horizontal():
    assert all(abs(a) < 1e-12 for a in inverse_kinematics(G3, 0.0, 0.0))


@pytest.mark.parametrize("tx, ty", [(10, 0), (0, 10), (-7, 4), (5, -9), (3, 3)])
def test_inverse_forward_round_trip(tx, ty):
    tx, ty = math.radians(tx), math.radians(ty)
    q = forward_kinematics(G3, inverse_kinematics(G3, tx, ty))
    assert abs(q[0] - tx) < 1e-9 and abs(q[1] - ty) < 1e-9 and abs(q[2] - G3.height) < 1e-9


def test_small_angle_transmission_ratio():
    # dα_i ≈ -(r/l1)·(cos φi·θx + sin φi·θy)
    tx, ty = math.radians(0.2), math.radians(-0.1)
    ratio = G3.attachment_radius / G3.arm_length
    for i, a in enumerate(inverse_kinematics(G3, tx, ty)):
        ux, uy = G3.direction(i)
        assert math.isclose(a, -ratio * (ux * tx + uy * ty), rel_tol=0.01, abs_tol=1e-6)


def test_unreachable_pose():
    with pytest.raises(ValueError):
        inverse_kinematics(G3, math.radians(60), 0.0)


def test_motor_bounded_speed_and_step():
    m = Motor(0.0, max_speed=2.0, resolution=0.01)
    m.request(1.004)                         # rounded to the step: 1.00
    for _ in range(100):
        m.advance(0.001)                      # 0.1 s at 2 rad/s
    assert math.isclose(m.position, 0.2, rel_tol=1e-9)
    for _ in range(1000):
        m.advance(0.001)
    assert math.isclose(m.position, 1.0, rel_tol=1e-9)


def test_motor_ramp_bounded_acceleration():
    m = Motor(0.0, max_speed=10.0, resolution=1e-4, max_acceleration=50.0)
    m.request(1.0)
    speeds = []
    for _ in range(600):
        m.advance(0.001)
        speeds.append(m.velocity)
    assert max(abs(b - a) / 0.001 for a, b in zip(speeds, speeds[1:])) <= 50.0 + 1e-6
    assert abs(m.position - 1.0) < 1e-3


def test_motor_two_ramps():
    # Large move (gap > fast_move): fast ramp until the motor has stopped on its target; small corrections:
    # gentle ramp. With fast_move = inf, a single gentle ramp
    def max_acceleration(m, target, steps):
        m.request(target)
        speeds = [m.velocity]
        for _ in range(steps):
            m.advance(0.001)
            speeds.append(m.velocity)
        return max(abs(b - a) / 0.001 for a, b in zip(speeds, speeds[1:]))

    m = Motor(0.0, max_speed=10.0, resolution=1e-4, max_acceleration=50.0, fast_acceleration=300.0, fast_move=0.1)
    assert math.isclose(max_acceleration(m, 0.5, 500), 300.0, rel_tol=1e-6)
    assert abs(m.position - 0.5) < 1e-3 and not m.fast       # stopped on its target: back to the gentle ramp
    assert max_acceleration(m, 0.55, 500) <= 50.0 + 1e-6      # small correction (gap 0.05 rad)
    single = Motor(0.0, max_speed=10.0, resolution=1e-4, max_acceleration=50.0, fast_acceleration=300.0)
    assert max_acceleration(single, 0.5, 1000) <= 50.0 + 1e-6
