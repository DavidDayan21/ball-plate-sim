"""Modèle physique de la balle sur une poutre (phase 1).

Convention de signe :
    x = position de la balle le long de la poutre (m), x = 0 au centre (pivot).
    θ = inclinaison de la poutre (rad). θ > 0 : le côté x > 0 est plus bas,
        donc la balle accélère vers les x positifs.

Équation du mouvement (voir le mini-cours) :
    ẍ = g·sin(θ) / (1 + k)      avec J = k·m·r² le moment d'inertie de la balle.

Hypothèses :
    - roulement sans glissement ;
    - angle θ imposé directement (on ne modélise pas encore les moteurs) ;
    - terme centrifuge m·x·θ̇² négligé (la poutre tourne lentement) ;
    - frottement de roulement et résistance de l'air négligés.
"""
from dataclasses import dataclass
import math

G = 9.81  # accélération de la pesanteur (m/s²)


@dataclass
class ParametresPoutre:
    longueur: float = 0.40       # longueur totale de la poutre (m)
    rayon_balle: float = 0.02    # rayon de la balle (m), utile plus tard (vision, contact)
    k_inertie: float = 2 / 3     # J = k·m·r² : 2/3 balle creuse (ping-pong), 2/5 bille pleine

    @property
    def coefficient(self) -> float:
        """Facteur 1/(1+k) de l'équation ẍ = g·sin(θ)/(1+k)."""
        return 1.0 / (1.0 + self.k_inertie)


def acceleration(theta: float, p: ParametresPoutre) -> float:
    """Accélération de la balle le long de la poutre (m/s²). θ en radians.

    La masse de la balle n'apparaît pas : elle se simplifie dans l'équation.
    """
    return p.coefficient * G * math.sin(theta)


def pas_euler(x: float, v: float, theta: float, dt: float, p: ParametresPoutre):
    """Un pas de la méthode d'Euler explicite.

    x_{n+1} = x_n + v_n·dt
    v_{n+1} = v_n + a(θ_n)·dt

    Renvoie (x, v) à l'instant t + dt.
    """
    a = acceleration(theta, p)
    return x + v * dt, v + a * dt


def pas_euler_semi_implicite(x: float, v: float, theta: float, dt: float, p: ParametresPoutre):
    """Un pas de la méthode d'Euler semi-implicite (dite symplectique).

    v_{n+1} = v_n + a(θ_n)·dt
    x_{n+1} = x_n + v_{n+1}·dt      <- utilise la NOUVELLE vitesse

    Même coût qu'Euler explicite, mais n'amplifie pas artificiellement les oscillations
    (voir le mini-cours de la phase 1b). C'est la méthode utilisée à partir de la phase 1b.

    Renvoie (x, v) à l'instant t + dt.
    """
    v = v + acceleration(theta, p) * dt
    return x + v * dt, v


def sur_la_poutre(x: float, p: ParametresPoutre) -> bool:
    """Vrai tant que la balle n'a pas dépassé une extrémité de la poutre."""
    return abs(x) <= p.longueur / 2
