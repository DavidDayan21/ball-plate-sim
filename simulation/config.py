"""Configuration unique de toutes les simulations, et trajectoires de consigne.

Tous les paramètres physiques sont des hypothèses de départ (aucun modèle réel de référence) :
ils peuvent être modifiés pour explorer d'autres conceptions.
"""
from dataclasses import dataclass, field
import math

from controller.cinematique import Geometrie

G = 9.81  # m/s²


# --- Paramètres physiques ------------------------------------------------------------------
@dataclass(frozen=True)
class ParametresPlateau:
    rayon_plateau: float = 0.15   # plateau circulaire (m)
    rayon_balle: float = 0.02     # (m)
    k_inertie: float = 2 / 3      # J = k·m·r² : 2/3 balle creuse (ping-pong), 2/5 bille pleine
    biais_x: float = 0.0          # défaut d'étalonnage sur θx (rad), inconnu du contrôleur
    biais_y: float = 0.0          # défaut d'étalonnage sur θy (rad), inconnu du contrôleur

    @property
    def coefficient(self) -> float:
        """c = 1/(1+k), dans ẍ = c·g·sin θ."""
        return 1.0 / (1.0 + self.k_inertie)


@dataclass(frozen=True)
class ParametresMujoco:
    """Paramètres ajoutés par la physique MuJoCo."""
    masse_balle: float = 0.0027        # balle de ping-pong (kg)
    masse_plateau: float = 0.25        # disque acrylique Ø30 cm, 3 mm (kg, estimation)
    epaisseur_plateau: float = 0.003   # (m)
    masse_bras: float = 0.02           # (kg)
    masse_biellette: float = 0.01      # (kg)
    kp_servo: float = 50.0             # raideur de l'asservissement de position (N·m/rad)
    kv_servo: float = 0.5              # amortissement de l'asservissement (N·m·s/rad)
    couple_max: float = 0.4            # couple max d'un moteur (N·m, ordre de grandeur d'un NEMA 17)
    armature: float = 5e-6             # inertie du rotor (kg·m²)
    solref_contact: str = "0.005 1"    # contact balle-plateau plus rigide que par défaut
    frottement: float = 1.0            # balle-plateau ; MuJoCo garde le plus grand des deux geoms,
                                       # on le met donc sur les deux
    k_reel: float = None               # inertie réelle de la balle ; None : celle que croit le contrôleur


@dataclass(frozen=True)
class ParametresCamera:
    hauteur: float = 0.50              # au-dessus du centre du plateau (m)
    fovy: float = 40.0                 # champ de vision vertical (°)
    largeur: int = 320                 # résolution (pixels)
    hauteur_image: int = 240
    teinte: tuple = (5, 35)            # plage de teinte OpenCV (0-179) de la balle orange
    saturation_min: int = 100
    valeur_min: int = 80
    pixels_min: int = 20               # en dessous : balle non détectée


# --- Trajectoires de consigne : t -> (x_ref, y_ref, ax_ref, ay_ref) -----------------------------
def reference(t: float):
    """Scénario de référence. 0-4 s : centre. 4-8 s : point (5 cm, 5 cm).
    Ensuite : cercle de 6 cm de rayon parcouru en 4 s, départ à 45°."""
    if t < 4.0:
        return 0.0, 0.0, 0.0, 0.0
    if t < 8.0:
        return 0.05, 0.05, 0.0, 0.0
    R, w = 0.06, 2 * math.pi / 4.0
    phi = w * (t - 8.0) + math.pi / 4
    x, y = R * math.cos(phi), R * math.sin(phi)
    return x, y, -w ** 2 * x, -w ** 2 * y


def centre(t: float):
    """Balle maintenue au centre."""
    return 0.0, 0.0, 0.0, 0.0


def carre(t: float):
    """Carré de 10 cm de côté à 5 cm/s (2 s par côté), départ au coin (5 cm, 5 cm)."""
    coins = [(0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05)]
    k, s = int(t // 2.0) % 4, (t % 2.0) / 2.0
    (x0, y0), (x1, y1) = coins[k], coins[(k + 1) % 4]
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 0.0, 0.0


# --- Configuration d'une simulation ------------------------------------------------------------
@dataclass
class Config:
    # Physique
    physique: str = "mujoco"           # "simple" (équations maison) ou "mujoco"
    mecanisme: bool = True             # True : 3 moteurs + cinématique ; False : angles appliqués directement
    plateau: ParametresPlateau = field(default_factory=lambda: ParametresPlateau(
        biais_x=math.radians(0.5), biais_y=math.radians(-0.3)))
    geometrie: Geometrie = field(default_factory=Geometrie)
    mujoco_params: ParametresMujoco = field(default_factory=ParametresMujoco)
    dt_physique: float = 0.001         # (s)
    duree: float = 20.0                # (s)
    # Scénario
    depart: tuple = (0.08, -0.06)      # position initiale de la balle (m)
    vitesse_depart: tuple = (0.0, 0.0)  # (m/s) : balle lancée
    poussees: tuple = ()               # chocs (t, Δvx, Δvy)
    trajectoire: object = reference    # fonction t -> (x_ref, y_ref, ax_ref, ay_ref)
    # Moteurs (si mecanisme)
    vitesse_max: float = 10.0          # (rad/s)
    resolution: float = math.radians(1.8 / 16)   # 200 pas/tour en 1/16 de pas
    acceleration_max: float = 50.0     # rampes (rad/s²) ; math.inf : sans rampe
    # Capteur
    capteur: str = "fictif"            # "fictif" (bruit gaussien) ou "camera" (MuJoCo uniquement)
    sigma: float = 0.001               # capteur fictif : écart-type du bruit (m)
    retard: int = 1                    # latence de mesure, en périodes de contrôle
    camera: ParametresCamera = field(default_factory=ParametresCamera)
    correction_inclinaison: bool = True
    # Contrôleur
    periode_controle: float = 0.02     # 50 Hz
    kp: float = 1.0                    # (rad/m)
    ki: float = 1.0                    # (rad/(m·s))
    zeta: float = 0.7                  # amortissement visé, donne Kd
    theta_max: float = math.radians(10)
    tau_d: float = 0.05                # filtre de la dérivée (s) ; 0 : pas de filtre
    anticipation: bool = True
    compensation_retard: bool = False  # prédire la position actuelle malgré la latence

    def kd(self) -> float:
        """Kd donnant l'amortissement ζ visé : 2·ζ·ω0 = c·g·Kd, avec ω0 = √(c·g·Kp)."""
        cg = self.plateau.coefficient * G
        return 2 * self.zeta * math.sqrt(cg * self.kp) / cg
