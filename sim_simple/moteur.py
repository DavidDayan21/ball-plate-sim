"""Modèle de moteur pas à pas asservi en position (phase 3).

Hypothèses : le moteur rejoint la position demandée à vitesse bornée, la position demandée
est arrondie au pas (ou micro-pas) le plus proche, et l'angle est limité par des butées.
Le couple et l'inertie ne sont pas modélisés (ils le seront implicitement dans MuJoCo).
"""
import math


class Moteur:
    def __init__(self, angle_initial: float, vitesse_max: float, resolution: float,
                 angle_min: float = math.radians(-60), angle_max: float = math.radians(60)):
        self.position = angle_initial     # angle réel (rad)
        self.cible = angle_initial        # angle demandé, arrondi au pas (rad)
        self.vitesse_max = vitesse_max    # rad/s
        self.resolution = resolution      # rad par pas
        self.angle_min, self.angle_max = angle_min, angle_max
        self.limite_vitesse = False       # vrai si le dernier pas a été limité en vitesse

    def demander(self, angle: float) -> None:
        """Nouvelle consigne d'angle : arrondie au pas le plus proche, puis bornée par les butées."""
        pas = round(angle / self.resolution) * self.resolution
        self.cible = min(max(pas, self.angle_min), self.angle_max)

    def avancer(self, dt: float) -> float:
        """Fait avancer le moteur pendant dt vers sa cible. Renvoie l'angle réel."""
        ecart = self.cible - self.position
        max_pas = self.vitesse_max * dt
        self.limite_vitesse = abs(ecart) > max_pas
        self.position += max(-max_pas, min(max_pas, ecart))
        return self.position
