"""Caméra virtuelle + détection de la balle par OpenCV.

Chaîne de mesure, identique à ce que ferait le vrai robot :
    image RGB -> seuillage de couleur (HSV) -> centre de la tache (moments) -> pixel (u, v)
    -> rayon lumineux du modèle sténopé -> intersection avec le plan du centre de la balle
    -> position (x, y) dans le repère du plateau

La caméra est fixe, au-dessus du plateau, et regarde vers le bas.
"""
import math

import cv2
import mujoco
import numpy as np

from controller.kinematics import rotation
from simulation.config import CameraParams


def inverse_rotation(theta_x: float, theta_y: float, v):
    """Applique R^T = Rx(θy)·Ry(-θx), l'inverse de la rotation du plateau."""
    x, y, z = v
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    x1, z1 = cx * x - sx * z, sx * x + cx * z           # Ry(-θx)
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    return x1, cy * y - sy * z1, sy * y + cy * z1       # Rx(θy)


class BallCamera:
    def __init__(self, model, cam: CameraParams, plate_height: float, ball_radius: float,
                 tilt_correction: bool = True):
        self.cam = cam
        self.renderer = mujoco.Renderer(model, cam.image_height, cam.width)
        # Modèle sténopé : focale en pixels f = (H/2) / tan(fovy/2), centre optique au milieu de l'image
        self.f = (cam.image_height / 2) / math.tan(math.radians(cam.fovy) / 2)
        self.cu, self.cv = cam.width / 2, cam.image_height / 2
        self.h, self.r = plate_height, ball_radius
        self.correction = tilt_correction
        self.undetected = 0

    def image(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera="top_camera")
        return self.renderer.render()

    def detect(self, img: np.ndarray):
        """Centre (u, v) de la balle en pixels (sous-pixel, par les moments), et le masque."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        c = self.cam
        mask = cv2.inRange(hsv, (c.hue[0], c.saturation_min, c.value_min), (c.hue[1], 255, 255))
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] < c.pixels_min:
            return None, mask
        # +0,5 : le pixel d'indice i couvre l'intervalle [i, i+1[, son centre est en i + 0,5
        return (m["m10"] / m["m00"] + 0.5, m["m01"] / m["m00"] + 0.5), mask

    def pixel_to_plate(self, u: float, v: float, theta_x: float, theta_y: float):
        """Position (x, y) de la balle dans le repère du plateau, à partir du pixel (u, v).

        Rayon issu de la caméra C = (0, 0, h + H) : d = ((u - cu)/f, -(v - cv)/f, -1).
        Le centre de la balle est dans le plan parallèle au plateau, à la distance r :
            n·(p - p0) = 0, avec n = R·ez la normale et p0 = (0, 0, h) + r·n.
        Intersection : p = C + s·d, avec s = n·(p0 - C) / (n·d).
        Sans correction d'inclinaison, on prend θx = θy = 0 (plateau supposé horizontal).
        """
        if not self.correction:
            theta_x = theta_y = 0.0
        d = np.array([(u - self.cu) / self.f, -(v - self.cv) / self.f, -1.0])
        C = np.array([0.0, 0.0, self.h + self.cam.height])
        n = np.array(rotation(theta_x, theta_y, (0.0, 0.0, 1.0)))
        p0 = np.array([0.0, 0.0, self.h]) + self.r * n
        p = C + (n @ (p0 - C)) / (n @ d) * d
        x, y, _ = inverse_rotation(theta_x, theta_y, p - np.array([0.0, 0.0, self.h]))
        return x, y

    def measure(self, data, theta_x: float, theta_y: float, last=(0.0, 0.0)):
        """Mesure complète. Si la balle n'est pas détectée, renvoie la dernière mesure connue."""
        center, _ = self.detect(self.image(data))
        if center is None:
            self.undetected += 1
            return last
        return self.pixel_to_plate(center[0], center[1], theta_x, theta_y)

    def close(self) -> None:
        self.renderer.close()
