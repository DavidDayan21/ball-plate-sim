"""Modèle de commande d'un moteur pas à pas asservi en position.

Hypothèses : la position demandée est arrondie au pas (ou micro-pas) le plus proche et bornée
par des butées ; le moteur la rejoint avec une vitesse bornée et, si acceleration_max est fini,
une accélération bornée (profil trapézoïdal, comme les drivers de moteurs pas à pas).

    acceleration_max = inf : profil « en créneau » (vitesse max immédiate)
    acceleration_max finie : rampes d'accélération et de freinage

Le couple et l'inertie ne sont pas modélisés ici : dans MuJoCo, ce profil sert de consigne
à l'asservissement du moteur, qui lui a un couple limité.
"""
import math


class Moteur:
    def __init__(self, angle_initial: float, vitesse_max: float, resolution: float,
                 angle_min: float = math.radians(-60), angle_max: float = math.radians(60),
                 acceleration_max: float = math.inf):
        self.position = angle_initial     # angle (rad)
        self.vitesse = 0.0                # vitesse (rad/s), utilisée avec les rampes
        self.cible = angle_initial        # angle demandé, arrondi au pas (rad)
        self.vitesse_max = vitesse_max    # rad/s
        self.acceleration_max = acceleration_max  # rad/s²
        self.resolution = resolution      # rad par pas
        self.angle_min, self.angle_max = angle_min, angle_max

    def demander(self, angle: float) -> None:
        """Nouvelle consigne d'angle : arrondie au pas le plus proche, puis bornée par les butées."""
        pas = round(angle / self.resolution) * self.resolution
        self.cible = min(max(pas, self.angle_min), self.angle_max)

    def avancer(self, dt: float) -> float:
        """Fait avancer le moteur pendant dt vers sa cible. Renvoie l'angle."""
        ecart = self.cible - self.position
        if math.isinf(self.acceleration_max):
            max_pas = self.vitesse_max * dt
            self.position += max(-max_pas, min(max_pas, ecart))
            return self.position
        # Rampe : on vise la plus grande vitesse qui permet encore de s'arrêter sur la cible
        # en freinant à a_max, soit v = √(2·a_max·|écart|), sans dépasser vitesse_max ;
        # et la vitesse ne change que de a_max·dt au plus par pas.
        v_voulue = math.copysign(min(self.vitesse_max, math.sqrt(2 * self.acceleration_max * abs(ecart))), ecart)
        dv = self.acceleration_max * dt
        self.vitesse += max(-dv, min(dv, v_voulue - self.vitesse))
        self.position += self.vitesse * dt
        return self.position
