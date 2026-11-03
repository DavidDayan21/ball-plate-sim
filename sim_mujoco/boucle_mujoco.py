"""Boucle fermée dans MuJoCo (phase 4).

Le contrôleur, le capteur, la cinématique inverse et le modèle de commande des moteurs sont
EXACTEMENT ceux des phases 2 et 3 : seule la physique change (MuJoCo au lieu des équations maison).

    4a (mecanisme=False) : plateau sur cardan, angles θ appliqués par deux servos
    4b (mecanisme=True)  : robot complet, angles moteurs issus de la cinématique inverse

Phase 5 : capteur="camera" remplace le capteur fictif par la caméra virtuelle + OpenCV.
"""
from dataclasses import dataclass, field
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from controller.cinematique import cinematique_inverse
from controller.plateau import ControleurPlateau
from sim_mujoco.camera import CameraBalle, ParametresCamera
from sim_mujoco.modeles import ParametresMujoco, xml_plateau, xml_robot
from sim_simple.boucle_plateau import kd_pour
from sim_simple.boucle_robot import ConfigRobot, ResultatsRobot
from sim_simple.capteur import Capteur
from sim_simple.moteur import Moteur
from sim_simple.plateau import G


@dataclass
class ConfigMujoco(ConfigRobot):
    mecanisme: bool = True                       # False : plateau commandé en angles (4a)
    acceleration_max: float = 50.0               # rampes des moteurs (rad/s²) ; inf : sans rampe
    mujoco_params: ParametresMujoco = field(default_factory=ParametresMujoco)
    # Phase 5 : capteur
    capteur: str = "fictif"                      # "fictif" (bruit gaussien) ou "camera" (image + OpenCV)
    camera: ParametresCamera = field(default_factory=ParametresCamera)
    correction_inclinaison: bool = True          # caméra : tenir compte de l'inclinaison du plateau
    # Phase 6
    compensation_retard: bool = False            # prédire la position actuelle malgré la latence


@dataclass
class ResultatsMujoco(ResultatsRobot):
    couples: list = field(default_factory=list)  # couples des moteurs (N·m), 4b uniquement
    sans_contact: list = field(default_factory=list)  # part des pas sans contact balle-plateau,
                                                       # sur chaque période de contrôle
    part_sans_contact: float = 0.0                     # sur toute la simulation
    erreur_mesure: list = field(default_factory=list)  # |mesure - vraie position| avant retard (m)
    non_detectee: int = 0                              # caméra : images sans balle détectée
    image_exemple: tuple = None                        # caméra : (image, masque, centre) vers t = 2 s


def construire(cfg: ConfigMujoco):
    if cfg.mecanisme:
        xml = xml_robot(cfg.geometrie, cfg.plateau, cfg.mujoco_params, cfg.depart, cfg.camera)
    else:
        xml = xml_plateau(cfg.plateau, cfg.mujoco_params, cfg.depart, cfg.geometrie.hauteur, cfg.camera)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Vitesse initiale (balle lancée), avec la rotation qui correspond au roulement sans glissement :
    # v + ω × (-r·ez) = 0  =>  ω = (-vy/r, vx/r, 0)
    vx, vy = cfg.vitesse_depart
    r = cfg.plateau.rayon_balle
    adr = model.jnt_dofadr[model.body("balle").jntadr[0]]
    data.qvel[adr:adr + 6] = (vx, vy, 0.0, -vy / r, vx / r, 0.0)
    mujoco.mj_forward(model, data)        # calcule les positions initiales (sinon xpos vaut 0)
    return model, data


def pousser(model, data, dvx: float, dvy: float, rayon: float) -> None:
    """Choc sur la balle : ajoute (Δvx, Δvy) à sa vitesse, avec la rotation de roulement associée."""
    adr = model.jnt_dofadr[model.body("balle").jntadr[0]]
    data.qvel[adr:adr + 2] += (dvx, dvy)
    # La vitesse de rotation d'une liberté (freejoint) s'exprime dans le repère de la balle
    R = data.body("balle").xmat.reshape(3, 3)
    data.qvel[adr + 3:adr + 6] += R.T @ np.array([-dvy / rayon, dvx / rayon, 0.0])


def balle_dans_repere_plateau(model, data):
    """Position (x, y, z) du centre de la balle dans le repère du plateau."""
    plateau = data.body("plateau")
    R = plateau.xmat.reshape(3, 3)
    return R.T @ (data.body("balle").xpos - plateau.xpos)


def inclinaison_plateau(data):
    """(θx, θy) réels du plateau, d'après sa normale n = R·ez = (sin θx cos θy, sin θy, cos θx cos θy)."""
    n = data.body("plateau").xmat.reshape(3, 3)[:, 2]
    return math.atan2(n[0], n[2]), math.asin(max(-1.0, min(1.0, n[1])))


def simuler(cfg: ConfigMujoco, vue: bool = False) -> ResultatsMujoco:
    model, data = construire(cfg)
    p, g = cfg.plateau, cfg.geometrie
    camera = None
    if cfg.capteur == "camera":
        camera = CameraBalle(model, cfg.camera, g.hauteur, p.rayon_balle, cfg.correction_inclinaison)
        capteur = Capteur(sigma=0.0, retard=cfg.retard)   # le bruit vient de l'image ; reste le retard
    else:
        capteur = Capteur(cfg.sigma, cfg.retard)
    ctrl = ControleurPlateau(cfg.kp, cfg.ki, kd_pour(cfg), cfg.theta_max, cfg.tau_d,
                             gain_modele=p.coefficient * G, anticipation=cfg.anticipation,
                             compensation_retard=cfg.retard if cfg.compensation_retard else 0,
                             periode=cfg.periode_controle)
    pas_poussees = {round(tp / cfg.dt_physique): (dvx, dvy) for tp, dvx, dvy in cfg.poussees}
    moteurs = ([Moteur(0.0, cfg.vitesse_max, cfg.resolution, acceleration_max=cfg.acceleration_max)
                for _ in range(3)] if cfg.mecanisme else [])
    pas_par_controle = round(cfg.periode_controle / cfg.dt_physique)
    n_limite = 0
    n_sans_contact = 0                    # total
    theta_x = theta_y = 0.0               # dernière commande (connue du robot, sert à la caméra)
    derniere_mesure = cfg.depart
    sans_contact_periode = 0              # sur la période de contrôle en cours
    r = ResultatsMujoco()

    viewer = None
    if vue:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.7, -30, 120
        viewer.cam.lookat[:] = (0, 0, g.hauteur)
        debut = time.perf_counter()

    n_pas = round(cfg.duree / cfg.dt_physique)
    for n in range(n_pas + 1):
        t = n * cfg.dt_physique           # pas data.time : la somme des dt accumule des arrondis
        bx, by, bz = balle_dans_repere_plateau(model, data)
        if math.hypot(bx, by) > p.rayon_plateau or bz < 0:
            r.tombee = True
            break

        if n % pas_par_controle == 0:
            if camera is not None:
                mx, my = camera.mesurer(data, theta_x, theta_y, derniere_mesure)
                derniere_mesure = (mx, my)
                if r.image_exemple is None and t >= 2.0:
                    img = camera.image(data)
                    centre, masque = camera.detecter(img)
                    r.image_exemple = (img, masque, centre)
            else:
                mx, my = bx, by
            xm, ym = capteur.mesurer(mx, my)
            if camera is None:            # capteur fictif : mesure bruitée de cet instant (avant retard)
                mx, my = capteur.file[-1] if capteur.file else (xm, ym)
            r.erreur_mesure.append(math.hypot(mx - bx, my - by))
            xr, yr, axr, ayr = cfg.trajectoire(t)
            theta_x, theta_y = ctrl.calculer((xr, yr), (xm, ym), (axr, ayr), cfg.periode_controle)
            # Défaut d'étalonnage : l'angle réellement visé = angle commandé + biais (inconnu du PID)
            cible_x, cible_y = theta_x + p.biais_x, theta_y + p.biais_y
            if cfg.mecanisme:
                try:
                    for m, a in zip(moteurs, cinematique_inverse(g, cible_x, cible_y)):
                        m.demander(a)
                except ValueError:
                    r.poses_hors_portee += 1
            else:
                data.ctrl[:] = (cible_x, cible_y)
            reel_x, reel_y = inclinaison_plateau(data)
            alphas = tuple(data.joint(f"moteur{i + 1}").qpos[0] for i in range(3)) if cfg.mecanisme else ()
            for liste, val in ((r.t, t), (r.x, bx), (r.y, by), (r.x_mes, xm), (r.y_mes, ym),
                               (r.x_ref, xr), (r.y_ref, yr), (r.theta_x, theta_x), (r.theta_y, theta_y),
                               (r.alphas, alphas), (r.theta_x_reel, reel_x - p.biais_x),
                               (r.theta_y_reel, reel_y - p.biais_y),
                               (r.couples, tuple(data.actuator_force) if cfg.mecanisme else ()),
                               (r.sans_contact, sans_contact_periode / pas_par_controle)):
                liste.append(val)
            sans_contact_periode = 0

        if n in pas_poussees:
            pousser(model, data, *pas_poussees[n], p.rayon_balle)
        if cfg.mecanisme:
            for i, m in enumerate(moteurs):
                data.ctrl[i] = m.avancer(cfg.dt_physique)
            n_limite += any(m.limite_vitesse for m in moteurs)
        mujoco.mj_step(model, data)
        if data.ncon == 0:                # seul le contact balle-plateau est activé dans le modèle
            n_sans_contact += 1
            sans_contact_periode += 1

        if viewer is not None and n % pas_par_controle == 0:
            if not viewer.is_running():
                break
            viewer.sync()
            retard = data.time - (time.perf_counter() - debut)
            if retard > 0:
                time.sleep(retard)            # cale la simulation sur le temps réel

    if viewer is not None:
        viewer.close()
    if camera is not None:
        r.non_detectee = camera.non_detectee
        camera.fermer()
    r.part_limite_vitesse = n_limite / max(1, n) if cfg.mecanisme else 0.0
    r.part_sans_contact = n_sans_contact / max(1, n)
    return r


def acceleration_roulement(k: float, theta: float, solref: str = "0.005 1", duree: float = 1.0,
                           frottement: float = 1.0) -> float:
    """Accélération mesurée par MuJoCo d'une balle (J = k·m·r²) sur un plan fixe incliné de θ.

    Sert à vérifier que MuJoCo retrouve ẍ = g·sin θ / (1 + k).
    """
    r, m = 0.02, 0.0027
    inertie = k * m * r * r
    xml = f"""<mujoco><option timestep="0.001" integrator="implicitfast"/><worldbody>
      <body euler="0 {math.degrees(theta)} 0"><geom type="box" size="0.5 0.5 0.005" pos="0 0 -0.005" friction="{frottement} 0.005 0.0001"/></body>
      <body name="balle"><freejoint/><geom type="sphere" size="{r}" solref="{solref}" friction="{frottement} 0.005 0.0001"/>
        <inertial pos="0 0 0" mass="{m}" diaginertia="{inertie} {inertie} {inertie}"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = np.array([math.sin(theta), 0.0, math.cos(theta)])        # normale au plan
    tangente = np.array([math.cos(theta), 0.0, -math.sin(theta)])  # direction de la pente
    data.qpos[:3] = n * r
    T, S = [], []
    while data.time < duree:
        mujoco.mj_step(model, data)
        T.append(data.time)
        S.append(float(data.qpos[:3] @ tangente))
    T, S = np.array(T), np.array(S)
    garde = T > 0.2                                              # on ignore la mise en contact
    return 2 * np.polyfit(T[garde], S[garde], 2)[0]               # s = ½·a·t² + ...
