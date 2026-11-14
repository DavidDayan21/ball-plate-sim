"""Gestures on the plate (app/gestures.py) and the camera that relates the screen to the plate (app/view3d.Camera).
Without Qt nor screen: pure computation, and Session for the round trips."""
import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("PySide6")          # app/view3d.py imports Qt (the camera itself does not use it)

from app.gestures import (ARROW_TIME, MAX_SPEED, drop_arguments, plate_hit, throw_arguments, throw_limit,
                          velocity)
from app.scene import DisplayScene
from app.theme import scene_style
from app.view3d import Camera
from simulation.config import G
from simulation.session import Session, sandbox_config


def test_camera_pixel_and_ray_agree():
    cam = Camera(0.1)
    cam.azimuth, cam.elevation, cam.distance = 37.0, -41.0, 0.7
    for point in ((0.0, 0.0, 0.1), (0.05, -0.08, 0.12), (-0.1, 0.04, 0.08)):
        x, y = cam.project(point, 800, 500)
        eye, direction = cam.ray(x, y, 800, 500)
        to_point = np.array(point) - eye
        assert np.allclose(np.cross(direction, to_point), 0, atol=1e-12)     # the ray goes through the point
    assert cam.project(cam.lookat, 800, 500) == pytest.approx((400, 250))     # the point looked at: the center
    cam.azimuth, cam.elevation = 90.0, -27.0                                  # the default view: x to the right
    (xr, _), (yr, yu) = cam.screen_axes()
    assert xr == pytest.approx(1.0) and yr == pytest.approx(0.0, abs=1e-12) and yu > 0


def test_camera_glides():
    cam = Camera(0.1)
    cam.glide((0.5, 0.0, 0.1), cam.distance + 1.0, dt=0.3, time_constant=0.3)
    assert cam.lookat[0] == pytest.approx(0.5 * (1 - math.exp(-1)))            # the gap shrinks by e^-1
    for _ in range(100):
        cam.glide((0.5, 0.0, 0.1), 1.68, dt=0.1, time_constant=0.3)
    assert cam.lookat == pytest.approx((0.5, 0.0, 0.1)) and cam.distance == pytest.approx(1.68)


def test_camera_matches_the_rendering():
    """The pixel computed for the center of the ball is where MuJoCo draws the ball (the orange of the image)."""
    s = Session(sandbox_config(), record=False)
    s.throw(**drop_arguments((0.06, -0.04), 0.0))
    scene = DisplayScene(s.cfg, scene_style("light"))
    scene.follow(s.phys.data)
    cam = Camera(s.cfg.geometry.height - 0.035, scene.model.vis.global_.fovy)
    cam.azimuth, cam.elevation = 60.0, -35.0
    width, height = 640, 400
    renderer = mujoco.Renderer(scene.model, height, width)
    renderer.update_scene(scene.data, camera=cam.mujoco())
    image = renderer.render().astype(int)
    renderer.close()
    orange = (image[..., 0] > 170) & (image[..., 1] > 50) & (image[..., 1] < 170) & (image[..., 2] < 90)
    ys, xs = np.nonzero(orange)
    x, y = cam.project(scene.data.body("ball").xpos, width, height)
    assert len(xs) > 100 and abs(xs.mean() - x) < 2 and abs(ys.mean() - y) < 2


def test_plate_hit():
    eye, down = np.array([0.03, -0.02, 0.5]), np.array([0.0, 0.0, -1.0])
    assert plate_hit(eye, down, (0, 0, 0.1), np.eye(3)) == pytest.approx((0.03, -0.02))
    assert plate_hit(eye, -down, (0, 0, 0.1), np.eye(3)) is None                     # the plate is behind
    tilt = math.radians(10)                                                          # plate tilted about y
    R = np.array([[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0], [-math.sin(tilt), 0, math.cos(tilt)]])
    x, y = plate_hit(np.array([0.0, 0.0, 0.5]), down, (0, 0, 0.1), R)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-12)
    x, _ = plate_hit(np.array([0.05, 0.0, 0.5]), down, (0, 0, 0.1), R)
    assert x == pytest.approx(0.05 / math.cos(tilt))                                 # along the slope


def test_velocity_of_a_gesture():
    assert velocity((0.0, 0.0), (0.1, 0.0)) == pytest.approx((0.1 / ARROW_TIME, 0.0))   # 10 cm: 0.5 m/s
    vx, vy = velocity((0.0, 0.0), (0.0, -2.0))
    assert (vx, vy) == pytest.approx((0.0, -MAX_SPEED))


def test_physical_limit_of_the_reference_throw():
    cfg = sandbox_config()
    limit = throw_limit((-0.10, 0.0), 1.0, 0.0, cfg.plate.plate_radius, cfg.theta_max, cfg.plate.coefficient * G)
    assert limit == pytest.approx(0.71, abs=0.005)             # results/report.md, section 2
    sideways = throw_limit((-0.10, 0.0), 0.0, 1.0, cfg.plate.plate_radius, cfg.theta_max,
                           cfg.plate.coefficient * G)
    assert sideways < limit                                    # less room toward the side


def test_throw_and_drop_round_trips():
    s = Session(sandbox_config(), record=False)
    start, (vx, vy) = (0.04, -0.07), (-0.2, 0.35)
    s.throw(**throw_arguments(start, vx, vy))
    assert s.phys.ball() == pytest.approx(start, abs=1e-9)
    assert s.phys.ball_velocity() == pytest.approx((vx, vy), abs=1e-9)
    s.throw(**throw_arguments((0.0, 0.0), 0.3, 0.0))           # from the center: any start angle
    assert s.phys.ball_velocity() == pytest.approx((0.3, 0.0), abs=1e-9)
    s.throw(**drop_arguments((-0.05, 0.02), 0.2))
    x, y, z = s.phys.ball_3d()
    assert (x, y) == pytest.approx((-0.05, 0.02), abs=1e-9)
    assert z == pytest.approx(0.2 + s.cfg.plate.ball_radius, abs=1e-9)
    assert s.attempt["action"] == "drop" and s.attempt["params"]["height_cm"] == pytest.approx(20)
