"""Boucle fermée complète : capteur -> PID -> cinématique inverse -> 3 moteurs
-> cinématique directe -> inclinaison réelle -> physique de la balle (phase 3).

Même scénario et mêmes indicateurs que la phase 2 (sim_simple/boucle_plateau.py).
"""
from dataclasses import dataclass, field
import math

from controller.cinematique import Geometrie, cinematique_directe, cinematique_inverse
from controller.plateau import ControleurPlateau
from sim_simple.boucle_plateau import Config, Resultats, kd_pour
from sim_simple.capteur import Capteur
from sim_simple.moteur import Moteur
from sim_simple.plateau import G, pas_semi_implicite, sur_le_plateau


@dataclass
class ConfigRobot(Config):
    geometrie: Geometrie = field(default_factory=Geometrie)
    moteurs: bool = True                          # False : plateau commandé directement (phase 2)
    vitesse_max: float = 10.0                     # vitesse max des moteurs (rad/s)
    resolution: float = math.radians(1.8 / 16)    # moteur 200 pas/tour en 1/16 de pas
    acceleration_max: float = math.inf            # rampes des moteurs (rad/s²), inf : pas de rampe


@dataclass
class ResultatsRobot(Resultats):
    alphas: list = field(default_factory=list)            # angles réels des moteurs (rad)
    theta_x_reel: list = field(default_factory=list)      # inclinaisons réelles (rad)
    theta_y_reel: list = field(default_factory=list)
    part_limite_vitesse: float = 0.0                       # fraction du temps en limite de vitesse
    poses_hors_portee: int = 0


def simuler(cfg: ConfigRobot) -> ResultatsRobot:
    p, g = cfg.plateau, cfg.geometrie
    capteur = Capteur(cfg.sigma, cfg.retard)
    ctrl = ControleurPlateau(cfg.kp, cfg.ki, kd_pour(cfg), cfg.theta_max, cfg.tau_d,
                             gain_modele=p.coefficient * G, anticipation=cfg.anticipation)
    alphas0 = cinematique_inverse(g, 0.0, 0.0)
    moteurs = [Moteur(a, cfg.vitesse_max, cfg.resolution, acceleration_max=cfg.acceleration_max)
               for a in alphas0]
    pose = (0.0, 0.0, g.hauteur)
    alphas = alphas0
    pas_par_controle = round(cfg.periode_controle / cfg.dt_physique)
    x, y = cfg.depart
    vx, vy = cfg.vitesse_depart
    theta_x = theta_y = 0.0
    n_limite = n_total = 0
    r = ResultatsRobot()

    for n in range(round(cfg.duree / cfg.dt_physique) + 1):
        t = n * cfg.dt_physique
        if n % pas_par_controle == 0:
            xm, ym = capteur.mesurer(x, y)
            xr, yr, axr, ayr = cfg.trajectoire(t)
            theta_x, theta_y = ctrl.calculer((xr, yr), (xm, ym), (axr, ayr), cfg.periode_controle)
            if cfg.moteurs:
                try:
                    for m, a in zip(moteurs, cinematique_inverse(g, theta_x, theta_y)):
                        m.demander(a)
                except ValueError:
                    r.poses_hors_portee += 1      # pose impossible : les moteurs gardent leur cible
            for liste, val in ((r.t, t), (r.x, x), (r.y, y), (r.x_mes, xm), (r.y_mes, ym),
                               (r.x_ref, xr), (r.y_ref, yr), (r.theta_x, theta_x), (r.theta_y, theta_y),
                               (r.alphas, alphas), (r.theta_x_reel, pose[0]), (r.theta_y_reel, pose[1])):
                liste.append(val)

        if cfg.moteurs:
            nouveaux = tuple(m.avancer(cfg.dt_physique) for m in moteurs)
            n_limite += any(m.limite_vitesse for m in moteurs)
            n_total += 1
            if nouveaux != alphas:                # la pose ne change que si un moteur a bougé
                alphas = nouveaux
                pose = cinematique_directe(g, alphas, depart=pose)
            tx_reel, ty_reel = pose[0], pose[1]
        else:
            tx_reel, ty_reel = theta_x, theta_y

        if not sur_le_plateau(x, y, p):
            r.tombee = True
            break
        x, y, vx, vy = pas_semi_implicite(x, y, vx, vy, tx_reel, ty_reel, cfg.dt_physique, p)

    r.part_limite_vitesse = n_limite / n_total if n_total else 0.0
    return r
