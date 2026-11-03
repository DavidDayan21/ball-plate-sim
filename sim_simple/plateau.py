"""Modèle physique de la balle sur un plateau 2D (phase 2).

Convention :
    (x, y) = position de la balle sur le plateau (m), (0, 0) au centre.
    θx = inclinaison du plateau dans la direction x (rad) : θx > 0 fait accélérer la balle vers +x.
    θy = idem dans la direction y.

Équations (modèle découplé, petits angles) : chaque axe se comporte comme la poutre de la phase 1.
    ẍ = c·g·sin(θx + bx)        ÿ = c·g·sin(θy + by)        c = 1 / (1 + k)

Hypothèses supplémentaires par rapport à la phase 1 :
    - couplage entre les deux axes négligé (il est d'ordre 2 en angle) ; MuJoCo le prendra
      en compte en phase 4 ;
    - plateau circulaire : la balle tombe si sa distance au centre dépasse rayon_plateau.
"""
from dataclasses import dataclass
import math

G = 9.81  # m/s²


@dataclass
class ParametresPlateau:
    rayon_plateau: float = 0.15   # rayon du plateau circulaire (m)
    rayon_balle: float = 0.02     # rayon de la balle (m)
    k_inertie: float = 2 / 3      # J = k·m·r² : 2/3 balle creuse (ping-pong), 2/5 bille pleine
    biais_x: float = 0.0          # défaut d'étalonnage sur θx (rad), inconnu du contrôleur
    biais_y: float = 0.0          # défaut d'étalonnage sur θy (rad), inconnu du contrôleur

    @property
    def coefficient(self) -> float:
        """Facteur c = 1/(1+k)."""
        return 1.0 / (1.0 + self.k_inertie)


def acceleration(theta_x: float, theta_y: float, p: ParametresPlateau):
    """Accélération (ax, ay) de la balle (m/s²) pour les angles commandés (θx, θy)."""
    cg = p.coefficient * G
    return cg * math.sin(theta_x + p.biais_x), cg * math.sin(theta_y + p.biais_y)


def pas_semi_implicite(x, y, vx, vy, theta_x, theta_y, dt, p: ParametresPlateau):
    """Un pas d'Euler semi-implicite. Renvoie (x, y, vx, vy) à t + dt."""
    ax, ay = acceleration(theta_x, theta_y, p)
    vx += ax * dt
    vy += ay * dt
    return x + vx * dt, y + vy * dt, vx, vy


def sur_le_plateau(x: float, y: float, p: ParametresPlateau) -> bool:
    return math.hypot(x, y) <= p.rayon_plateau
