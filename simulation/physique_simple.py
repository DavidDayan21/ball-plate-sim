"""Physique « maison » : nos propres équations, sans moteur physique externe.

Balle sur plateau, modèle découplé (petits angles), intégré par Euler semi-implicite :
    ẍ = c·g·sin(θx + bx)        ÿ = c·g·sin(θy + by)        c = 1 / (1 + k)

Hypothèses : roulement sans glissement, contact permanent, couplage entre axes et effets de la
rotation du plateau négligés. Avec mecanisme=True, les angles passent par les 3 moteurs
(simulation/moteur.py) et l'inclinaison réelle est donnée par la cinématique directe.

Sert de référence théorique : ses résultats se démontrent à la main (voir tests/test_controleur.py).
"""
import math

from controller.cinematique import cinematique_directe, cinematique_inverse
from simulation.config import G, Config, ParametresPlateau
from simulation.moteur import Moteur


def acceleration(theta_x: float, theta_y: float, p: ParametresPlateau):
    """Accélération (ax, ay) de la balle (m/s²) pour les angles commandés (θx, θy)."""
    cg = p.coefficient * G
    return cg * math.sin(theta_x + p.biais_x), cg * math.sin(theta_y + p.biais_y)


def pas_semi_implicite(x, y, vx, vy, theta_x, theta_y, dt, p: ParametresPlateau):
    """Un pas d'Euler semi-implicite : la nouvelle vitesse sert à calculer la nouvelle position."""
    ax, ay = acceleration(theta_x, theta_y, p)
    vx += ax * dt
    vy += ay * dt
    return x + vx * dt, y + vy * dt, vx, vy


class PhysiqueSimple:
    def __init__(self, cfg: Config):
        self.p, self.g = cfg.plateau, cfg.geometrie
        self.mecanisme = cfg.mecanisme
        self.x, self.y = cfg.depart
        self.vx, self.vy = cfg.vitesse_depart
        self.commande = (0.0, 0.0)
        self.poses_hors_portee = 0
        if self.mecanisme:
            self.alphas = cinematique_inverse(self.g, 0.0, 0.0)
            self.moteurs = [Moteur(a, cfg.vitesse_max, cfg.resolution, acceleration_max=cfg.acceleration_max)
                            for a in self.alphas]
            self.pose = (0.0, 0.0, self.g.hauteur)

    def balle(self):
        """Position (x, y) de la balle dans le repère du plateau (m)."""
        return self.x, self.y

    def tombee(self) -> bool:
        return math.hypot(self.x, self.y) > self.p.rayon_plateau

    def commander(self, theta_x: float, theta_y: float) -> None:
        self.commande = (theta_x, theta_y)
        if self.mecanisme:
            try:
                for m, a in zip(self.moteurs, cinematique_inverse(self.g, theta_x, theta_y)):
                    m.demander(a)
            except ValueError:
                self.poses_hors_portee += 1     # pose impossible : les moteurs gardent leur cible

    def inclinaison_reelle(self):
        return (self.pose[0], self.pose[1]) if self.mecanisme else self.commande

    def angles_moteurs(self):
        return self.alphas if self.mecanisme else ()

    def vitesse_balle(self):
        return self.vx, self.vy

    def hauteur_balle(self) -> float:
        return 0.0                              # modèle plan : la balle reste au contact

    def pousser(self, dvx: float, dvy: float) -> None:
        self.vx += dvx
        self.vy += dvy

    def placer_balle(self, x, y, z=0.0, vx=0.0, vy=0.0, vz=0.0) -> None:
        """Téléporte la balle (repère du plateau). Le modèle plan ignore la hauteur z et vz."""
        self.x, self.y, self.vx, self.vy = x, y, vx, vy

    def pas(self, dt: float) -> None:
        if self.mecanisme:
            nouveaux = tuple(m.avancer(dt) for m in self.moteurs)
            if nouveaux != self.alphas:          # la pose ne change que si un moteur a bougé
                self.alphas = nouveaux
                self.pose = cinematique_directe(self.g, self.alphas, depart=self.pose)
        tx, ty = self.inclinaison_reelle()
        self.x, self.y, self.vx, self.vy = pas_semi_implicite(self.x, self.y, self.vx, self.vy,
                                                              tx, ty, dt, self.p)
