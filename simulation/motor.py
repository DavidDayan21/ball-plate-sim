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


class Motor:
    def __init__(self, initial_angle: float, max_speed: float, resolution: float,
                 angle_min: float = math.radians(-60), angle_max: float = math.radians(60),
                 max_acceleration: float = math.inf):
        self.position = initial_angle     # angle (rad)
        self.velocity = 0.0                # vitesse (rad/s), utilisée avec les rampes
        self.target = initial_angle        # angle demandé, arrondi au pas (rad)
        self.max_speed = max_speed    # rad/s
        self.max_acceleration = max_acceleration  # rad/s²
        self.resolution = resolution      # rad par pas
        self.angle_min, self.angle_max = angle_min, angle_max

    def request(self, angle: float) -> None:
        """Nouvelle consigne d'angle : arrondie au pas le plus proche, puis bornée par les butées."""
        quantized = round(angle / self.resolution) * self.resolution
        self.target = min(max(quantized, self.angle_min), self.angle_max)

    def advance(self, dt: float) -> float:
        """Fait avancer le moteur pendant dt vers sa cible. Renvoie l'angle."""
        gap = self.target - self.position
        if math.isinf(self.max_acceleration):
            max_step = self.max_speed * dt
            self.position += max(-max_step, min(max_step, gap))
            return self.position
        # Rampe : on vise la plus grande vitesse qui permet encore de s'arrêter sur la cible
        # en freinant à a_max, soit v = √(2·a_max·|écart|), sans dépasser vitesse_max ;
        # et la vitesse ne change que de a_max·dt au plus par pas.
        v_target = math.copysign(min(self.max_speed, math.sqrt(2 * self.max_acceleration * abs(gap))), gap)
        dv = self.max_acceleration * dt
        self.velocity += max(-dv, min(dv, v_target - self.velocity))
        self.position += self.velocity * dt
        return self.position
