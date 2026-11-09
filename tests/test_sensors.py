"""Synthetic sensor and virtual camera (MuJoCo rendering + OpenCV detection)."""
from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from controller.kinematics import rotation
from simulation.camera import BallCamera, inverse_rotation
from simulation.sensor import Sensor
from simulation.config import Config, CameraParams, center
from simulation.loop import simulate
from simulation.mujoco_physics import model_xml

H, R = 0.12, 0.02


def test_sensor_exact_latency():
    # Like a real camera, the first measurement arrives only after the latency
    c = Sensor(sigma=0.0, latency=2)
    measures = [c.measure(float(k), 0.0) for k in range(6)]
    assert measures[:2] == [None, None] and [m[0] for m in measures[2:]] == [0.0, 1.0, 2.0, 3.0]


def test_height_noise_has_its_own_draws():
    # The noise on the height comes from its own generator: the position noise stays the same
    a, b = Sensor(sigma=0.001, latency=0, seed=3), Sensor(sigma=0.001, latency=0, seed=3, sigma_height=0.003)
    ma, mb = [a.measure(0.0, 0.0, 0.05) for _ in range(2000)], [b.measure(0.0, 0.0, 0.05) for _ in range(2000)]
    assert [m[:2] for m in ma] == [m[:2] for m in mb] and all(m[2] == 0.05 for m in ma)
    assert np.std([m[2] for m in mb]) == pytest.approx(0.003, rel=0.1)


def test_measured_height_recorded():
    r = simulate(Config(trajectory=center, start_position=(0.0, 0.0), start_height=0.10, duration=0.1))
    assert abs(r.measured_height[0] - 0.10) < 0.012 and abs(r.measured_height[0] - r.height[0]) < 0.012


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
    cameras["airborne"] = BallCamera(model, CameraParams(), H, R, airborne_height=0.015)
    yield model, data, cameras
    for c in cameras.values():
        c.close()


def place(scene, tx, ty, x, y, z=0.0):
    """Ball at (x, y), z above the plate tilted by (tx, ty) (rad)."""
    model, data, _ = scene
    data.qpos[:] = 0
    data.joint("tilt_x").qpos[0] = tx
    data.joint("tilt_y").qpos[0] = ty
    adr = model.jnt_qposadr[model.body("ball").jntadr[0]]
    data.qpos[adr:adr + 3] = np.array(rotation(tx, ty, (x, y, R + z))) + np.array([0, 0, H])
    data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)


def measure(scene, tx_deg, ty_deg, x, y, correction=True):
    model, data, cameras = scene
    tx, ty = math.radians(tx_deg), math.radians(ty_deg)
    place(scene, tx, ty, x, y)
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


@pytest.mark.parametrize("z", [0.0, 0.02, 0.05, 0.10])
@pytest.mark.parametrize("x, y", [(0.0, 0.0), (0.05, -0.03), (-0.08, 0.06)])
def test_camera_measures_the_height(scene, x, y, z):
    # Height from the apparent size of the ball; in the air, its center is taken on the light ray
    place(scene, 0.0, 0.0, x, y, z)
    cam = scene[2]["airborne"]
    xm, ym, hm = cam.measure(scene[1], 0.0, 0.0)
    assert abs(hm - z) < 0.008 and math.hypot(xm - x, ym - y) < 0.006


def test_height_on_a_tilted_plate(scene):
    # At rest on the plate tilted by 10°, the ball is 1.7 cm closer to the camera: not in the air
    tx, ty = math.radians(10), math.radians(-10)
    place(scene, tx, ty, 0.10, 0.10)
    xm, ym, hm = scene[2]["airborne"].measure(scene[1], tx, ty)
    assert abs(hm) < 0.008 and math.hypot(xm - 0.10, ym - 0.10) < 0.0015


def test_perspective_of_a_ball_in_the_air(scene):
    # 10 cm above the plate, the ball looks 2.6 cm farther from the center if taken on the plane of the plate
    place(scene, 0.0, 0.0, 0.08, -0.06, 0.10)
    on_ray, on_plane = scene[2]["airborne"].measure(scene[1], 0.0, 0.0), scene[2][True].measure(scene[1], 0.0, 0.0)
    assert math.hypot(on_ray[0] - 0.08, on_ray[1] + 0.06) < 0.005
    assert math.hypot(on_plane[0] - 0.08, on_plane[1] + 0.06) > 0.02


def test_ball_partly_out_of_the_image_not_measured(scene):
    place(scene, 0.0, 0.0, 0.10, 0.08, 0.30)            # 30 cm high near the edge: cut by the edge of the image
    cam = scene[2]["airborne"]
    assert cam.detect(cam.image(scene[1]))[0] is None
    assert cam.measure(scene[1], 0.0, 0.0, last=(1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_flight_seen_by_the_camera():
    # Drop from 20 cm, 5.8 cm from the center, level plate: camera 160×120, 20 ms of latency, ballistic
    # estimator (controller/flight.py). The impact predicted 40 ms before is right to a few ms and mm.
    from controller.flight import FlightEstimator
    from simulation.config import PlateParams
    from simulation.mujoco_physics import MujocoPhysics
    cfg = Config(mechanism=False, plate=PlateParams(), start_position=(0.05, -0.03), start_height=0.20)
    phys = MujocoPhysics(cfg)
    cam = BallCamera(phys.model, CameraParams(width=160, image_height=120), H, R, airborne_height=0.015)
    sensor, est = Sensor(sigma=0.0, latency=1), FlightEstimator(H, R, 0.02, 1, 0.015)
    est.command((0.0, 0.0))
    last, predictions, n, dt = (0.05, -0.03, 0.20), [], 0, cfg.dt_physics
    while phys.data.ncon == 0:
        if n % 40 == 0:
            last = cam.measure(phys.data, 0.0, 0.0, last)
            measured = sensor.measure(*last)
            if measured is not None:
                est.observe(measured)
                est.command((0.0, 0.0))
            if measured is not None and est.impact() is not None:
                predictions.append((n * dt, est.impact()))
        vz = phys.data.qvel[phys._ball_address()[1] + 2]       # vertical speed just before the contact
        phys.command(0.0, 0.0)
        phys.step(dt)
        n += 1
    cam.close()
    t_impact = n * dt
    now, (time_left, p, v) = [pr for pr in predictions if pr[0] <= t_impact - 0.04][-1]
    assert abs(now + time_left - t_impact) < 0.003
    assert math.hypot(p[0] - 0.05, p[1] + 0.03) < 0.002 and v[2] == pytest.approx(vz, rel=0.02)
