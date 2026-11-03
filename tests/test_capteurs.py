"""Capteur fictif et caméra virtuelle (rendu MuJoCo + détection OpenCV)."""
import math

import mujoco
import numpy as np
import pytest

from controller.cinematique import rotation
from simulation.camera import CameraBalle, rotation_inverse
from simulation.capteur import Capteur
from simulation.config import Config, ParametresCamera
from simulation.physique_mujoco import xml_modele

H, R = 0.12, 0.02


def test_capteur_retard_exact():
    c = Capteur(sigma=0.0, retard=2)
    assert [c.mesurer(float(k), 0.0)[0] for k in range(6)] == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_rotation_inverse():
    v = (0.1, -0.2, 0.3)
    w = rotation_inverse(0.3, -0.2, rotation(0.3, -0.2, v))
    assert all(abs(a - b) < 1e-12 for a, b in zip(v, w))


@pytest.fixture(scope="module")
def scene():
    """Plateau en angles + caméra : on place la balle à la main pour tester la mesure."""
    model = mujoco.MjModel.from_xml_string(xml_modele(Config(mecanisme=False, depart=(0.0, 0.0))))
    data = mujoco.MjData(model)
    cameras = {c: CameraBalle(model, ParametresCamera(), H, R, correction_inclinaison=c) for c in (True, False)}
    yield model, data, cameras
    for c in cameras.values():
        c.fermer()


def mesurer(scene, tx_deg, ty_deg, x, y, correction=True):
    model, data, cameras = scene
    tx, ty = math.radians(tx_deg), math.radians(ty_deg)
    data.qpos[:] = 0
    data.joint("inclinaison_x").qpos[0] = tx
    data.joint("inclinaison_y").qpos[0] = ty
    adr = model.jnt_qposadr[model.body("balle").jntadr[0]]
    data.qpos[adr:adr + 3] = np.array(rotation(tx, ty, (x, y, R))) + np.array([0, 0, H])
    data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    cam = cameras[correction]
    centre, _ = cam.detecter(cam.image(data))
    xm, ym = cam.pixel_vers_plateau(*centre, tx, ty)
    return math.hypot(xm - x, ym - y)


@pytest.mark.parametrize("x, y", [(0.0, 0.0), (0.05, -0.03), (-0.08, 0.06), (0.12, 0.08)])
def test_camera_precise_plateau_horizontal(scene, x, y):
    assert mesurer(scene, 0, 0, x, y) < 0.001


def test_correction_inclinaison_indispensable_aux_grands_angles(scene):
    assert mesurer(scene, 10, -10, 0.10, 0.10, correction=True) < 0.0015
    assert mesurer(scene, 10, -10, 0.10, 0.10, correction=False) > 0.005
