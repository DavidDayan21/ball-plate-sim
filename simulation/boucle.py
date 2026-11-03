"""Boucle fermée unique, quelle que soit la physique.

À chaque pas (0,5 ms) : la physique avance. Toutes les 20 ms (50 Hz) :
    mesure (capteur fictif ou caméra) -> contrôleur -> commande (θx, θy) -> moteurs / servos

    physique = "simple" : nos équations (simulation/physique_simple.py)
    physique = "mujoco" : moteur physique 3D (simulation/physique_mujoco.py)
"""
from dataclasses import dataclass, field
import math
import time

from controller.plateau import ControleurPlateau
from simulation.capteur import Capteur
from simulation.config import G, Config


@dataclass
class Resultats:
    """Historique complet, échantillonné à chaque période de contrôle (données d'une tentative)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)                    # vraie position de la balle (m)
    y: list = field(default_factory=list)
    x_ref: list = field(default_factory=list)                # consigne (m)
    y_ref: list = field(default_factory=list)
    theta_x: list = field(default_factory=list)              # commandes du contrôleur (rad)
    theta_y: list = field(default_factory=list)
    inclinaison_reelle: list = field(default_factory=list)   # (θx, θy) réels du plateau (rad)
    alphas: list = field(default_factory=list)               # angles des 3 moteurs (rad)
    vitesse: list = field(default_factory=list)              # (vx, vy) de la balle, repère du plateau (m/s)
    hauteur: list = field(default_factory=list)              # hauteur de la balle au-dessus du plateau (m)
    couples: list = field(default_factory=list)              # couples des 3 moteurs (N·m), MuJoCo
    decollee: list = field(default_factory=list)             # part de la période balle décollée, MuJoCo
    erreur_mesure: list = field(default_factory=list)        # |mesure - vraie position| avant latence (m)
    tombee: bool = False
    poses_hors_portee: int = 0
    part_decollee: float = 0.0
    non_detectee: int = 0                                    # caméra : images sans balle détectée


def creer_physique(cfg: Config):
    if cfg.physique == "simple":
        from simulation.physique_simple import PhysiqueSimple
        return PhysiqueSimple(cfg)
    if cfg.physique == "mujoco":
        from simulation.physique_mujoco import PhysiqueMujoco
        return PhysiqueMujoco(cfg)
    raise ValueError(f"Physique inconnue : {cfg.physique}")


class Simulation:
    """Simulation pas à pas (un pas = dt_physique).

    simuler() l'enchaîne sur toute la durée du scénario ; le bac à sable la pilote en direct.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.phys = creer_physique(cfg)
        self.camera = None
        if cfg.capteur == "camera":
            self.creer_camera()
        self.reinitialiser_controle()
        self.pas_poussees = {round(tp / cfg.dt_physique): (dvx, dvy) for tp, dvx, dvy in cfg.poussees}
        self.pas_par_controle = round(cfg.periode_controle / cfg.dt_physique)
        self.theta = (0.0, 0.0)             # dernière commande (connue du robot, sert à la caméra)
        self.derniere_mesure = cfg.depart
        self.n = 0                          # numéro du prochain pas
        self.r = Resultats()

    def creer_camera(self) -> None:
        """Caméra virtuelle. Son contexte OpenGL n'est utilisable que dans le fil (thread) qui l'a
        créée : la simulation doit tourner dans ce même fil."""
        from simulation.camera import CameraBalle
        cfg = self.cfg
        self.camera = CameraBalle(self.phys.model, cfg.camera, cfg.geometrie.hauteur,
                                  cfg.plateau.rayon_balle, cfg.correction_inclinaison)

    def reinitialiser_controle(self) -> None:
        """(Re)crée le capteur et le contrôleur à partir de self.cfg, mémoire remise à zéro."""
        cfg = self.cfg
        if self.camera is not None:
            self.capteur = Capteur(sigma=0.0, retard=cfg.retard)   # le bruit vient de l'image
        else:
            self.capteur = Capteur(cfg.sigma, cfg.retard)
        self.ctrl = ControleurPlateau(cfg.kp, cfg.ki, cfg.kd(), cfg.theta_max, cfg.tau_d,
                                      gain_modele=cfg.plateau.coefficient * G, anticipation=cfg.anticipation,
                                      compensation_retard=cfg.retard if cfg.compensation_retard else 0,
                                      periode=cfg.periode_controle)

    @property
    def t(self) -> float:
        return self.n * self.cfg.dt_physique    # pas de somme de dt : elle accumulerait des arrondis

    def pas(self) -> bool:
        """Avance d'un pas. Renvoie False si la balle est tombée."""
        cfg, phys, r, n = self.cfg, self.phys, self.r, self.n
        t = self.t
        if phys.tombee():
            r.tombee = True
            return False

        if n % self.pas_par_controle == 0:
            bx, by = phys.balle()
            if self.camera is not None:
                mx, my = self.camera.mesurer(phys.data, *self.theta, self.derniere_mesure)
                self.derniere_mesure = (mx, my)
            else:
                mx, my = bx, by
            xm, ym = self.capteur.mesurer(mx, my)
            if self.camera is None:         # capteur fictif : mesure bruitée de cet instant (avant latence)
                mx, my = self.capteur.file[-1] if self.capteur.file else (xm, ym)
            r.erreur_mesure.append(math.hypot(mx - bx, my - by))
            xr, yr, axr, ayr = cfg.trajectoire(t)
            self.theta = self.ctrl.calculer((xr, yr), (xm, ym), (axr, ayr), cfg.periode_controle)
            phys.commander(*self.theta)
            for liste, val in ((r.t, t), (r.x, bx), (r.y, by), (r.x_ref, xr), (r.y_ref, yr),
                               (r.theta_x, self.theta[0]), (r.theta_y, self.theta[1]),
                               (r.inclinaison_reelle, phys.inclinaison_reelle()),
                               (r.alphas, phys.angles_moteurs()),
                               (r.vitesse, phys.vitesse_balle()), (r.hauteur, phys.hauteur_balle())):
                liste.append(val)
            if cfg.physique == "mujoco":
                r.couples.append(phys.couples())
                r.decollee.append(phys.lire_decolles_periode() / self.pas_par_controle)

        if n in self.pas_poussees:
            phys.pousser(*self.pas_poussees[n])
        phys.pas(cfg.dt_physique)
        self.n += 1
        return True

    def terminer(self) -> Resultats:
        """Libère les ressources et complète les indicateurs globaux."""
        r = self.r
        if self.camera is not None:
            r.non_detectee = self.camera.non_detectee
            self.camera.fermer()
        r.poses_hors_portee = self.phys.poses_hors_portee
        if self.cfg.physique == "mujoco":
            dernier = self.n if r.tombee else self.n - 1
            r.part_decollee = self.phys.pas_decolles / max(1, dernier)
        return r


def simuler(cfg: Config, vue: bool = False) -> Resultats:
    """Simule le scénario de cfg. vue=True (MuJoCo) : fenêtre 3D en temps réel."""
    sim = Simulation(cfg)
    viewer = None
    if vue:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.phys.model, sim.phys.data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.7, -30, 120
        viewer.cam.lookat[:] = (0, 0, cfg.geometrie.hauteur)
        debut = time.perf_counter()

    for n in range(round(cfg.duree / cfg.dt_physique) + 1):
        if not sim.pas():
            break
        if viewer is not None and n % sim.pas_par_controle == 0:
            if not viewer.is_running():
                break
            viewer.sync()
            avance = sim.phys.data.time - (time.perf_counter() - debut)
            if avance > 0:
                time.sleep(avance)          # cale la simulation sur le temps réel

    if viewer is not None:
        viewer.close()
    return sim.terminer()
