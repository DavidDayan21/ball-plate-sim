"""Virtual camera + ball detection with OpenCV.

Measurement chain, identical to what the real robot would do:
    RGB image -> color thresholding (HSV) -> center of the blob (moments) -> pixel (u, v)
    -> light ray of the pinhole model -> intersection with the plane of the ball center
    -> position (x, y) in the plate frame
The area of the blob gives the distance of the ball, hence its height above the plate: a ball in the air
looks bigger. Above airborne_height, (x, y) is taken on the ray at that distance instead of on the plane
(otherwise a ball in the air would look farther from the center than it is: perspective).

The camera is fixed, above the plate, and looks down.
"""
import math

import cv2
import mujoco
import numpy as np

from controller.kinematics import rotation
from simulation.config import CameraParams


def inverse_rotation(theta_x: float, theta_y: float, v):
    """Applies R^T = Rx(θy)·Ry(-θx), the inverse of the plate rotation."""
    x, y, z = v
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    x1, z1 = cx * x - sx * z, sx * x + cx * z           # Ry(-θx)
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    return x1, cy * y - sy * z1, sy * y + cy * z1       # Rx(θy)


class BallCamera:
    def __init__(self, model, cam: CameraParams, plate_height: float, ball_radius: float,
                 tilt_correction: bool = True, airborne_height: float = math.inf):
        self.cam = cam
        self.renderer = mujoco.Renderer(model, cam.image_height, cam.width)
        # Pinhole model: focal length in pixels f = (H/2) / tan(fovy/2), optical center in the middle of the image
        self.f = (cam.image_height / 2) / math.tan(math.radians(cam.fovy) / 2)
        self.cu, self.cv = cam.width / 2, cam.image_height / 2
        self.h, self.r = plate_height, ball_radius
        self.correction = tilt_correction
        self.airborne_height = airborne_height
        self.undetected = 0
        self.area_ratio = self.calibrate(model)

    def calibrate(self, model) -> float:
        """Detected area / geometric area of the ball (the color threshold loses part of the edge pixels),
        averaged over images of the ball at rest on the level plate at 24 points 3 to 9 cm from the center,
        as one would calibrate the real camera."""
        data = mujoco.MjData(model)
        adr = model.jnt_qposadr[model.body("ball").jntadr[0]]
        ratios = []
        for radius in (0.03, 0.06, 0.09):
            for k in range(8):
                x, y = radius * math.cos(k * math.pi / 4), radius * math.sin(k * math.pi / 4)
                data.qpos[adr:adr + 3] = (x, y, self.h + self.r)
                data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_forward(model, data)
                _, area = self.detect(self.image(data))
                distance = math.sqrt(x * x + y * y + (self.cam.height - self.r) ** 2)
                ratios.append(area / self.geometric_area(distance, (self.cam.height - self.r) / distance))
        return sum(ratios) / len(ratios)

    def geometric_area(self, distance: float, cos_alpha: float) -> float:
        """Area in pixels of the image of a sphere of radius r whose center is at `distance` from the camera,
        seen at the angle α from the optical axis: π·f²·tan²β / cos³α, with sin β = r / distance
        (ellipse stretched away from the axis; exact on the axis, first order in β elsewhere)."""
        beta = math.asin(self.r / distance)
        return math.pi * (self.f * math.tan(beta)) ** 2 / cos_alpha ** 3

    def image(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera="top_camera")
        return self.renderer.render()

    def detect(self, img: np.ndarray):
        """Center (u, v) of the ball in pixels (sub-pixel, from the moments), and its area in pixels.
        None if the ball is not seen, or not entirely (cut by the edge of the image: wrong center and area)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        c = self.cam
        mask = cv2.inRange(hsv, (c.hue[0], c.saturation_min, c.value_min), (c.hue[1], 255, 255))
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] < c.pixels_min or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            return None, m["m00"]
        # +0.5: the pixel with index i covers the interval [i, i+1[, its center is at i + 0.5
        return (m["m10"] / m["m00"] + 0.5, m["m01"] / m["m00"] + 0.5), m["m00"]

    def pixel_to_plate(self, u: float, v: float, theta_x: float, theta_y: float):
        """Position (x, y) of the ball in the plate frame, from the pixel (u, v).

        Ray from the camera C = (0, 0, h + H): d = ((u - cu)/f, -(v - cv)/f, -1).
        The ball center lies in the plane parallel to the plate, at the distance r:
            n·(p - p0) = 0, with n = R·ez the normal and p0 = (0, 0, h) + r·n.
        Intersection: p = C + s·d, with s = n·(p0 - C) / (n·d).
        Without tilt correction, θx = θy = 0 is used (plate assumed horizontal).
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

    def height(self, u: float, v: float, area: float, theta_x: float, theta_y: float):
        """(height of the bottom of the ball above the plate, center of the ball in the plate frame), from the
        area of its image: tan²β = area·cos³α / (area_ratio·π·f²), distance = r / sin β along the ray."""
        if not self.correction:
            theta_x = theta_y = 0.0
        d = np.array([(u - self.cu) / self.f, -(v - self.cv) / self.f, -1.0])
        norm = float(np.linalg.norm(d))
        cos_alpha = 1.0 / norm
        beta = math.atan(math.sqrt(area * cos_alpha ** 3 / (self.area_ratio * math.pi)) / self.f)
        center = np.array([0.0, 0.0, self.h + self.cam.height]) + self.r / math.sin(beta) * d / norm
        p = inverse_rotation(theta_x, theta_y, center - np.array([0.0, 0.0, self.h]))
        return p[2] - self.r, p

    def measure(self, data, theta_x: float, theta_y: float, last=(0.0, 0.0, 0.0)):
        """Full measurement (x, y, height of the ball above the plate).
        If the ball is not detected, returns the last known measurement."""
        center, area = self.detect(self.image(data))
        if center is None:
            self.undetected += 1
            return last
        x, y = self.pixel_to_plate(center[0], center[1], theta_x, theta_y)
        height, p = self.height(center[0], center[1], area, theta_x, theta_y)
        if height > self.airborne_height:             # ball in the air: its center is on the ray, not on the plane
            x, y = p[0], p[1]
        return x, y, height

    def close(self) -> None:
        self.renderer.close()
