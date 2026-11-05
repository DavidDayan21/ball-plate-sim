"""Configuration unique de toutes les simulations, et trajectoires de consigne.

Tous les paramètres physiques sont des hypothèses de départ (aucun modèle réel de référence) :
ils peuvent être modifiés pour explorer d'autres conceptions.
"""
from dataclasses import dataclass, field
import math

from controller.kinematics import Geometry

G = 9.81  # m/s²


# --- Paramètres physiques ------------------------------------------------------------------
@dataclass(frozen=True)
class PlateParams:
    plate_radius: float = 0.15   # plateau circulaire (m)
    ball_radius: float = 0.02     # (m)
    k_inertia: float = 2 / 3      # J = k·m·r² : 2/3 balle creuse (ping-pong), 2/5 bille pleine
    bias_x: float = 0.0          # défaut d'étalonnage sur θx (rad), inconnu du contrôleur
    bias_y: float = 0.0          # défaut d'étalonnage sur θy (rad), inconnu du contrôleur

    @property
    def coefficient(self) -> float:
        """c = 1/(1+k), dans ẍ = c·g·sin θ."""
        return 1.0 / (1.0 + self.k_inertia)


@dataclass(frozen=True)
class MujocoParams:
    """Paramètres ajoutés par la physique MuJoCo."""
    ball_mass: float = 0.0027        # balle de ping-pong (kg)
    plate_mass: float = 0.25        # disque acrylique Ø30 cm, 3 mm (kg, estimation)
    plate_thickness: float = 0.003   # (m)
    arm_mass: float = 0.02           # (kg)
    rod_mass: float = 0.01      # (kg)
    kp_servo: float = 50.0             # raideur de l'asservissement de position (N·m/rad)
    kv_servo: float = 0.5              # amortissement de l'asservissement (N·m·s/rad)
    max_torque: float = 0.4            # couple max d'un moteur (N·m, ordre de grandeur d'un NEMA 17)
    armature: float = 5e-6             # inertie du rotor (kg·m²)
    restitution: float = 0.88          # coefficient de restitution balle-plateau visé. Règle ITTF :
                                       # lâchée de 30,5 cm, la balle remonte de 22 à 25 cm, soit
                                       # e = √(h2/h1) entre 0,85 et 0,91 (pertes dans l'air incluses)
    contact_frequency: float = 200.0   # raideur du contact, exprimée en fréquence propre (Hz).
                                       # Exige dt_physique ≤ 0,5 ms (sinon rebonds parasites)
    friction: float = 1.0            # balle-plateau ; MuJoCo garde le plus grand des deux geoms,
                                       # on le met donc sur les deux
    k_actual: float | None = None        # inertie réelle de la balle ; None : celle que croit le contrôleur


@dataclass(frozen=True)
class CameraParams:
    height: float = 0.50              # au-dessus du centre du plateau (m)
    fovy: float = 40.0                 # champ de vision vertical (°)
    width: int = 320                 # résolution (pixels)
    image_height: int = 240
    hue: tuple = (5, 35)            # plage de teinte OpenCV (0-179) de la balle orange
    saturation_min: int = 100
    value_min: int = 80
    pixels_min: int = 20               # en dessous : balle non détectée


# --- Trajectoires de consigne : t -> (x_ref, y_ref, ax_ref, ay_ref) -----------------------------
def reference(t: float):
    """Scénario de référence. 0-4 s : centre. 4-8 s : point (5 cm, 5 cm).
    Ensuite : cercle de 6 cm de rayon parcouru en 4 s, départ à 45°."""
    if t < 4.0:
        return 0.0, 0.0, 0.0, 0.0
    if t < 8.0:
        return 0.05, 0.05, 0.0, 0.0
    return circle(t - 8.0)


def center(t: float):
    """Balle maintenue au centre."""
    return 0.0, 0.0, 0.0, 0.0


def circle(t: float):
    """Cercle de 6 cm de rayon parcouru en 4 s (vitesse 9,4 cm/s), départ à 45°."""
    R, w = 0.06, 2 * math.pi / 4.0
    phi = w * t + math.pi / 4
    x, y = R * math.cos(phi), R * math.sin(phi)
    return x, y, -w ** 2 * x, -w ** 2 * y


def square(t: float):
    """Carré de 10 cm de côté à 5 cm/s (2 s par côté), départ au coin (5 cm, 5 cm)."""
    corners = [(0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05)]
    k, s = int(t // 2.0) % 4, (t % 2.0) / 2.0
    (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 0.0, 0.0


# --- Configuration d'une simulation ------------------------------------------------------------
@dataclass
class Config:
    # Physique
    physics: str = "mujoco"           # "simple" (équations maison) ou "mujoco"
    mechanism: bool = True             # True : 3 moteurs + cinématique ; False : angles appliqués directement
    plate: PlateParams = field(default_factory=lambda: PlateParams(
        bias_x=math.radians(0.5), bias_y=math.radians(-0.3)))
    geometry: Geometry = field(default_factory=Geometry)
    mujoco_params: MujocoParams = field(default_factory=MujocoParams)
    dt_physics: float = 0.0005        # (s) : 0,5 ms, imposé par le contact élastique
    duration: float = 20.0                # (s)
    # Scénario
    start_position: tuple = (0.08, -0.06)      # position initiale de la balle (m)
    start_velocity: tuple = (0.0, 0.0)  # (m/s) : balle lancée
    pushes: tuple = ()               # chocs (t, Δvx, Δvy)
    trajectory: object = reference    # fonction t -> (x_ref, y_ref, ax_ref, ay_ref)
    # Moteurs (si mecanisme)
    max_speed: float = 10.0          # (rad/s)
    resolution: float = math.radians(1.8 / 16)   # 200 pas/tour en 1/16 de pas
    max_acceleration: float = 50.0     # rampes (rad/s²) ; math.inf : sans rampe
    # Capteur
    sensor: str = "synthetic"            # "fictif" (bruit gaussien) ou "camera" (MuJoCo uniquement)
    sigma: float = 0.001               # capteur fictif : écart-type du bruit (m)
    latency: int = 1                    # latence de mesure, en périodes de contrôle
    camera: CameraParams = field(default_factory=CameraParams)
    tilt_correction: bool = True
    # Contrôleur
    control_period: float = 0.02     # 50 Hz
    kp: float = 1.0                    # (rad/m)
    ki: float = 1.0                    # (rad/(m·s))
    zeta: float = 0.7                  # amortissement visé, donne Kd
    theta_max: float = math.radians(10)
    tau_d: float = 0.05                # filtre de la dérivée (s) ; 0 : pas de filtre
    feedforward: bool = True
    latency_compensation: bool = False  # prédire la position actuelle malgré la latence

    def kd(self) -> float:
        """Kd donnant l'amortissement ζ visé : 2·ζ·ω0 = c·g·Kd, avec ω0 = √(c·g·Kp)."""
        cg = self.plate.coefficient * G
        return 2 * self.zeta * math.sqrt(cg * self.kp) / cg
