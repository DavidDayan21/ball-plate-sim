"""Boucle fermée du plateau 2D : physique (1 ms) + capteur + contrôleur (50 Hz).

Partagée par le script de la phase 2 et par les tests automatiques.
"""
from dataclasses import dataclass, field
import math

from controller.plateau import ControleurPlateau
from sim_simple.capteur import Capteur
from sim_simple.plateau import G, ParametresPlateau, pas_semi_implicite, sur_le_plateau


@dataclass
class Config:
    # Physique
    plateau: ParametresPlateau = field(default_factory=lambda: ParametresPlateau(
        biais_x=math.radians(0.5), biais_y=math.radians(-0.3)))
    dt_physique: float = 0.001      # pas de la simulation physique (s)
    duree: float = 20.0             # durée simulée (s)
    depart: tuple = (0.08, -0.06)   # position initiale de la balle, au repos (m)
    # Capteur
    sigma: float = 0.001            # écart-type du bruit de mesure (m)
    retard: int = 1                 # retard de mesure, en périodes de contrôle
    # Contrôleur
    periode_controle: float = 0.02  # 50 Hz
    kp: float = 1.0                 # rad/m
    ki: float = 1.0                 # rad/(m·s)
    zeta: float = 0.7               # amortissement visé, donne Kd
    theta_max: float = math.radians(10)
    tau_d: float = 0.05             # filtre de la dérivée (s) ; 0 = pas de filtre
    anticipation: bool = True


def consigne(t: float):
    """Trajectoire voulue : (x_ref, y_ref, ax_ref, ay_ref).

    0-4 s : centre. 4-8 s : point (5 cm, 5 cm). À partir de 8 s : cercle de 6 cm de rayon,
    parcouru en 4 s, qui démarre à 45° (près du point précédent).
    """
    if t < 4.0:
        return 0.0, 0.0, 0.0, 0.0
    if t < 8.0:
        return 0.05, 0.05, 0.0, 0.0
    R, w = 0.06, 2 * math.pi / 4.0
    phi = w * (t - 8.0) + math.pi / 4
    x, y = R * math.cos(phi), R * math.sin(phi)
    return x, y, -w ** 2 * x, -w ** 2 * y


@dataclass
class Resultats:
    """Historique, échantillonné à la fréquence du contrôleur."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    x_mes: list = field(default_factory=list)
    y_mes: list = field(default_factory=list)
    x_ref: list = field(default_factory=list)
    y_ref: list = field(default_factory=list)
    theta_x: list = field(default_factory=list)
    theta_y: list = field(default_factory=list)
    tombee: bool = False


def kd_pour(cfg: Config) -> float:
    """Kd donnant l'amortissement ζ visé : 2·ζ·ω0 = c·g·Kd, avec ω0 = √(c·g·Kp)."""
    cg = cfg.plateau.coefficient * G
    return 2 * cfg.zeta * math.sqrt(cg * cfg.kp) / cg


def simuler(cfg: Config) -> Resultats:
    p = cfg.plateau
    capteur = Capteur(cfg.sigma, cfg.retard)
    ctrl = ControleurPlateau(cfg.kp, cfg.ki, kd_pour(cfg), cfg.theta_max, cfg.tau_d,
                             gain_modele=p.coefficient * G, anticipation=cfg.anticipation)
    pas_par_controle = round(cfg.periode_controle / cfg.dt_physique)
    x, y = cfg.depart
    vx = vy = 0.0
    theta_x = theta_y = 0.0
    r = Resultats()

    for n in range(round(cfg.duree / cfg.dt_physique) + 1):
        t = n * cfg.dt_physique
        if n % pas_par_controle == 0:
            # Le contrôleur ne s'exécute qu'à 50 Hz ; entre deux calculs, θ reste constant
            xm, ym = capteur.mesurer(x, y)
            xr, yr, axr, ayr = consigne(t)
            theta_x, theta_y = ctrl.calculer((xr, yr), (xm, ym), (axr, ayr), cfg.periode_controle)
            for liste, val in ((r.t, t), (r.x, x), (r.y, y), (r.x_mes, xm), (r.y_mes, ym),
                               (r.x_ref, xr), (r.y_ref, yr), (r.theta_x, theta_x), (r.theta_y, theta_y)):
                liste.append(val)
        if not sur_le_plateau(x, y, p):
            r.tombee = True
            break
        x, y, vx, vy = pas_semi_implicite(x, y, vx, vy, theta_x, theta_y, cfg.dt_physique, p)
    return r


# --- Indicateurs ---------------------------------------------------------------------
def indices(r: Resultats, t0: float, t1: float):
    return [i for i, t in enumerate(r.t) if t0 <= t < t1]


def erreur_suivi_rms(r: Resultats, t0: float, t1: float) -> float:
    """Écart quadratique moyen entre la balle et la consigne sur [t0, t1[ (m)."""
    idx = indices(r, t0, t1)
    return math.sqrt(sum((r.x[i] - r.x_ref[i]) ** 2 + (r.y[i] - r.y_ref[i]) ** 2 for i in idx) / len(idx))


def temps_reponse_5(r: Resultats, t1: float) -> float:
    """Temps de réponse à 5 % de la distance initiale, pour le retour au centre (avant t1)."""
    d0 = math.hypot(r.x[0], r.y[0])
    idx = indices(r, 0.0, t1)
    for i in reversed(idx):
        if math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) > 0.05 * d0:
            return r.t[i]
    return 0.0


def ecart_type_theta(r: Resultats, t0: float, t1: float) -> float:
    """Écart-type de θx sur [t0, t1[ (rad) : mesure l'agitation de la commande."""
    vals = [r.theta_x[i] for i in indices(r, t0, t1)]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
