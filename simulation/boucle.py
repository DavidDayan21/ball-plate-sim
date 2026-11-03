"""Boucle fermée unique, quelle que soit la physique.

À chaque pas de 1 ms : la physique avance. Toutes les 20 ms (50 Hz) :
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
    couples: list = field(default_factory=list)              # couples des 3 moteurs (N·m), MuJoCo
    sans_contact: list = field(default_factory=list)         # part de la période sans contact, MuJoCo
    erreur_mesure: list = field(default_factory=list)        # |mesure - vraie position| avant latence (m)
    tombee: bool = False
    poses_hors_portee: int = 0
    part_sans_contact: float = 0.0
    non_detectee: int = 0                                    # caméra : images sans balle détectée


def creer_physique(cfg: Config):
    if cfg.physique == "simple":
        from simulation.physique_simple import PhysiqueSimple
        return PhysiqueSimple(cfg)
    if cfg.physique == "mujoco":
        from simulation.physique_mujoco import PhysiqueMujoco
        return PhysiqueMujoco(cfg)
    raise ValueError(f"Physique inconnue : {cfg.physique}")


def simuler(cfg: Config, vue: bool = False) -> Resultats:
    """Simule le scénario de cfg. vue=True (MuJoCo) : fenêtre 3D en temps réel."""
    phys = creer_physique(cfg)
    p = cfg.plateau
    camera = None
    if cfg.capteur == "camera":
        from simulation.camera import CameraBalle
        camera = CameraBalle(phys.model, cfg.camera, cfg.geometrie.hauteur, p.rayon_balle,
                             cfg.correction_inclinaison)
        capteur = Capteur(sigma=0.0, retard=cfg.retard)   # le bruit vient de l'image ; reste la latence
    else:
        capteur = Capteur(cfg.sigma, cfg.retard)
    ctrl = ControleurPlateau(cfg.kp, cfg.ki, cfg.kd(), cfg.theta_max, cfg.tau_d,
                             gain_modele=p.coefficient * G, anticipation=cfg.anticipation,
                             compensation_retard=cfg.retard if cfg.compensation_retard else 0,
                             periode=cfg.periode_controle)
    pas_poussees = {round(tp / cfg.dt_physique): (dvx, dvy) for tp, dvx, dvy in cfg.poussees}
    pas_par_controle = round(cfg.periode_controle / cfg.dt_physique)
    theta_x = theta_y = 0.0                 # dernière commande (connue du robot, sert à la caméra)
    derniere_mesure = cfg.depart
    r = Resultats()

    viewer = None
    if vue:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(phys.model, phys.data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.7, -30, 120
        viewer.cam.lookat[:] = (0, 0, cfg.geometrie.hauteur)
        debut = time.perf_counter()

    n_pas = round(cfg.duree / cfg.dt_physique)
    for n in range(n_pas + 1):
        t = n * cfg.dt_physique             # pas de somme de dt : elle accumulerait des arrondis
        if phys.tombee():
            r.tombee = True
            break

        if n % pas_par_controle == 0:
            bx, by = phys.balle()
            if camera is not None:
                mx, my = camera.mesurer(phys.data, theta_x, theta_y, derniere_mesure)
                derniere_mesure = (mx, my)
            else:
                mx, my = bx, by
            xm, ym = capteur.mesurer(mx, my)
            if camera is None:              # capteur fictif : mesure bruitée de cet instant (avant latence)
                mx, my = capteur.file[-1] if capteur.file else (xm, ym)
            r.erreur_mesure.append(math.hypot(mx - bx, my - by))
            xr, yr, axr, ayr = cfg.trajectoire(t)
            theta_x, theta_y = ctrl.calculer((xr, yr), (xm, ym), (axr, ayr), cfg.periode_controle)
            phys.commander(theta_x, theta_y)
            for liste, val in ((r.t, t), (r.x, bx), (r.y, by), (r.x_ref, xr), (r.y_ref, yr),
                               (r.theta_x, theta_x), (r.theta_y, theta_y),
                               (r.inclinaison_reelle, phys.inclinaison_reelle()),
                               (r.alphas, phys.angles_moteurs())):
                liste.append(val)
            if cfg.physique == "mujoco":
                r.couples.append(phys.couples())
                r.sans_contact.append(phys.lire_sans_contact_periode() / pas_par_controle)

        if n in pas_poussees:
            phys.pousser(*pas_poussees[n])
        phys.pas(cfg.dt_physique)

        if viewer is not None and n % pas_par_controle == 0:
            if not viewer.is_running():
                break
            viewer.sync()
            avance = phys.data.time - (time.perf_counter() - debut)
            if avance > 0:
                time.sleep(avance)          # cale la simulation sur le temps réel

    if viewer is not None:
        viewer.close()
    if camera is not None:
        r.non_detectee = camera.non_detectee
        camera.fermer()
    r.poses_hors_portee = phys.poses_hors_portee
    if cfg.physique == "mujoco":
        r.part_sans_contact = phys.pas_sans_contact / max(1, n)
    return r
