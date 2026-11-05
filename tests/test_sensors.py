"""Synthetic sensor and virtual camera (MuJoCo rendering + OpenCV detection)."""
from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from controller.kinematics import rotation
from simulation.camera import BallCamera, inverse_rotation
from simulation.sensor import Sensor
from simulation.config import Config, CameraParams
from simulation.loop import simulate
from simulation.mujoco_physics import model_xml

H, R = 0.12, 0.02


def test_sensor_exact_latency():
    c = Sensor(sigma=0.0, latency=2)
    assert [c.measure(float(k), 0.0)[0] for k in range(6)] == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_noise_seed_set_by_config():
    # Same seed: identical results; another seed: other noise, hence another trajectory
    cfg = Config(physics="simple", mechanism=False, duration=1.0)
    a, b, c = simulate(cfg), simulate(cfg), simulate(replace(cfg, seed=1))
    assert a.x == b.x and a.measurement_error == b.measurement_error
    assert c.measurement_error != a.measurement_error and c.x != a.x


def test_inverse_rotation():
    v = (0.1, -0.2, 0.3)
    w = inverse_rotation(0.3, -0.2, rotation(0.3, -0.2, v))
    assert all(abs(a - b) < 1e-12 for a, b in zip(v, w))


@pytest.fixture(scope="module")
def scene():
    """Angle-driven plate + camera: the ball is placed by hand to test the measurement."""
    model = mujoco.MjModel.from_xml_string(model_xml(Config(mechanism=False, start_position=(0.0, 0.0))))
    data = mujoco.MjData(model)
    cameras = {c: BallCamera(model, CameraParams(), H, R, tilt_correction=c) for c in (True, False)}
    yield model, data, cameras
    for c in cameras.values():
        c.close()


def measure(scene, tx_deg, ty_deg, x, y, correction=True):
    model, data, cameras = scene
    tx, ty = math.radians(tx_deg), math.radians(ty_deg)
    data.qpos[:] = 0
    data.joint("tilt_x").qpos[0] = tx
    data.joint("tilt_y").qpos[0] = ty
    adr = model.jnt_qposadr[model.body("ball").jntadr[0]]
    data.qpos[adr:adr + 3] = np.array(rotation(tx, ty, (x, y, R))) + np.array([0, 0, H])
    data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    cam = cameras[correction]
    center, _ = cam.detect(cam.image(data))
    xm, ym = cam.pixel_to_plate(*center, tx, ty)
    return math.hypot(xm - x, ym - y)


@pytest.mark.parametrize("x, y", [(0.0, 0.0), (0.05, -0.03), (-0.08, 0.06), (0.12, 0.08)])
def test_camera_accurate_level_plate(scene, x, y):
    assert measure(scene, 0, 0, x, y) < 0.001


def test_tilt_correction_required_at_large_angles(scene):
    assert measure(scene, 10, -10, 0.10, 0.10, correction=True) < 0.0015
    assert measure(scene, 10, -10, 0.10, 0.10, correction=False) > 0.005
