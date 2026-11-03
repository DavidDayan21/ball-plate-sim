"""Physique MuJoCo : modèles 3D (générés depuis la configuration) et interface de simulation.

Deux modèles, avec une caméra fixe « camera_haut » au-dessus du plateau :
    - mecanisme=False : plateau sur cardan (2 articulations), angles appliqués par deux servos
    - mecanisme=True  : robot complet, 3 × (moteur -> bras -> coude -> biellette), plateau libre ;
      chaque biellette est reliée au plateau par une contrainte « connect » (rotule), qui ferme
      la chaîne : 6 + 3 + 3 - 3×3 = 3 degrés de liberté, pilotés par les 3 moteurs.

Exporter les modèles pour les ouvrir dans le visualiseur MuJoCo :
    python -m simulation.physique_mujoco
    python -m mujoco.viewer --mjcf=results/robot.xml
"""
import math
import os

import mujoco
import numpy as np

from controller.cinematique import cinematique_inverse
from simulation.config import Config
from simulation.moteur import Moteur

# --- Génération des modèles (format MJCF) ------------------------------------------------------
ENTETE = """<mujoco model="{nom}">
  <option timestep="{dt}" integrator="implicitfast"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <visual><global azimuth="120" elevation="-30"/></visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="sol" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.04" rgba="0.85 0.85 0.85 1"/>
    <camera name="camera_haut" pos="0 0 {z_camera}" xyaxes="1 0 0 0 1 0" fovy="{fovy}"/>"""


def _balle(cfg: Config) -> str:
    pp, pm = cfg.plateau, cfg.mujoco_params
    r = pp.rayon_balle
    k = pp.k_inertie if pm.k_reel is None else pm.k_reel
    inertie = k * pm.masse_balle * r * r                 # J = k·m·r², donnée explicitement
    x0, y0 = cfg.depart
    return f"""
    <body name="balle" pos="{x0} {y0} {cfg.geometrie.hauteur + r}">
      <freejoint/>
      <geom name="balle" type="sphere" size="{r}" contype="1" conaffinity="1"
            solref="{pm.solref_contact}" friction="{pm.frottement} 0.005 0.0001" rgba="1 0.5 0 1"/>
      <inertial pos="0 0 0" mass="{pm.masse_balle}" diaginertia="{inertie} {inertie} {inertie}"/>
    </body>"""


def _plateau_geom(cfg: Config) -> str:
    pp, pm = cfg.plateau, cfg.mujoco_params
    e = pm.epaisseur_plateau
    return (f'<geom name="plateau" type="cylinder" size="{pp.rayon_plateau} {e / 2}" pos="0 0 {-e / 2}" '
            f'mass="{pm.masse_plateau}" contype="1" conaffinity="1" friction="{pm.frottement} 0.005 0.0001" '
            f'rgba="0.7 0.85 1 0.5"/>')


def xml_modele(cfg: Config) -> str:
    g, pm = cfg.geometrie, cfg.mujoco_params
    h = g.hauteur
    entete = ENTETE.format(nom="robot equilibriste" if cfg.mecanisme else "plateau en angles",
                           dt=cfg.dt_physique, z_camera=h + cfg.camera.hauteur, fovy=cfg.camera.fovy)
    if not cfg.mecanisme:
        # Articulations : θx (axe y), puis θy (axe -x) : rotation R = Ry(θx)·Rx(-θy)
        return entete + f"""
    <geom name="pied" type="cylinder" size="0.01 {h / 2 + 0.02}" pos="0 0 {h / 2 - 0.02}" rgba="0.3 0.3 0.3 1"/>
    <body name="cardan" pos="0 0 {h}">
      <joint name="inclinaison_x" type="hinge" axis="0 1 0" range="-30 30"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
      <body name="plateau">
        <joint name="inclinaison_y" type="hinge" axis="-1 0 0" range="-30 30"/>
        {_plateau_geom(cfg)}
      </body>
    </body>{_balle(cfg)}
  </worldbody>
  <actuator>
    <position name="servo_x" joint="inclinaison_x" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
    <position name="servo_y" joint="inclinaison_y" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
  </actuator>
</mujoco>
"""
    l1, l2 = g.longueur_bras, g.longueur_biellette
    jambes, contraintes, actionneurs = "", "", ""
    for i in range(3):
        ux, uy = g.direction(i)
        bx, by = g.rayon_base * ux, g.rayon_base * uy
        axe = f"{uy} {-ux} 0"            # u × z : un angle α > 0 lève le bras
        jambes += f"""
    <geom name="support{i + 1}" type="box" size="0.012 0.012 0.02" pos="{bx} {by} -0.02" rgba="0.2 0.2 0.2 1"/>
    <body name="bras{i + 1}" pos="{bx} {by} 0">
      <joint name="moteur{i + 1}" type="hinge" axis="{axe}" range="-60 60" armature="{pm.armature}"/>
      <geom type="capsule" fromto="0 0 0 {l1 * ux} {l1 * uy} 0" size="0.006" mass="{pm.masse_bras}" rgba="0.2 0.4 0.8 1"/>
      <body name="biellette{i + 1}" pos="{l1 * ux} {l1 * uy} 0">
        <joint name="coude{i + 1}" type="hinge" axis="{axe}" damping="1e-4"/>
        <geom type="capsule" fromto="0 0 0 0 0 {l2}" size="0.004" mass="{pm.masse_biellette}" rgba="0.55 0.55 0.6 1"/>
      </body>
    </body>"""
        contraintes += f'\n    <connect body1="biellette{i + 1}" body2="plateau" anchor="0 0 {l2}" solref="0.005 1"/>'
        actionneurs += (f'\n    <position name="servo{i + 1}" joint="moteur{i + 1}" kp="{pm.kp_servo}" '
                        f'kv="{pm.kv_servo}" forcerange="{-pm.couple_max} {pm.couple_max}"/>')
    return entete + jambes + f"""
    <body name="plateau" pos="0 0 {h}">
      <freejoint/>
      {_plateau_geom(cfg)}
    </body>{_balle(cfg)}
  </worldbody>
  <equality>{contraintes}
  </equality>
  <actuator>{actionneurs}
  </actuator>
</mujoco>
"""


def construire(cfg: Config):
    """Modèle + données MuJoCo, balle placée avec sa vitesse initiale."""
    model = mujoco.MjModel.from_xml_string(xml_modele(cfg))
    data = mujoco.MjData(model)
    # Vitesse initiale avec la rotation du roulement sans glissement : v + ω × (-r·ez) = 0
    vx, vy = cfg.vitesse_depart
    r = cfg.plateau.rayon_balle
    adr = model.jnt_dofadr[model.body("balle").jntadr[0]]
    data.qvel[adr:adr + 6] = (vx, vy, 0.0, -vy / r, vx / r, 0.0)
    mujoco.mj_forward(model, data)        # calcule les positions initiales (sinon xpos vaut 0)
    return model, data


def inclinaison_plateau(data):
    """(θx, θy) du plateau, d'après sa normale n = R·ez = (sin θx cos θy, sin θy, cos θx cos θy)."""
    n = data.body("plateau").xmat.reshape(3, 3)[:, 2]
    return math.atan2(n[0], n[2]), math.asin(max(-1.0, min(1.0, n[1])))


# --- Interface de simulation (même interface que PhysiqueSimple) -----------------------------------
class PhysiqueMujoco:
    def __init__(self, cfg: Config):
        self.cfg, self.p, self.g = cfg, cfg.plateau, cfg.geometrie
        self.mecanisme = cfg.mecanisme
        self.model, self.data = construire(cfg)
        self.moteurs = ([Moteur(0.0, cfg.vitesse_max, cfg.resolution, acceleration_max=cfg.acceleration_max)
                         for _ in range(3)] if self.mecanisme else [])
        self.poses_hors_portee = 0
        self.pas_sans_contact = 0             # total
        self.pas_sans_contact_periode = 0     # depuis la dernière lecture

    def balle_3d(self):
        """Centre de la balle dans le repère du plateau (x, y, z)."""
        plateau = self.data.body("plateau")
        R = plateau.xmat.reshape(3, 3)
        return R.T @ (self.data.body("balle").xpos - plateau.xpos)

    def balle(self):
        bx, by, _ = self.balle_3d()
        return bx, by

    def tombee(self) -> bool:
        bx, by, bz = self.balle_3d()
        return math.hypot(bx, by) > self.p.rayon_plateau or bz < 0

    def commander(self, theta_x: float, theta_y: float) -> None:
        # Défaut d'étalonnage : l'angle réellement visé = angle commandé + biais (inconnu du PID)
        cible_x, cible_y = theta_x + self.p.biais_x, theta_y + self.p.biais_y
        if self.mecanisme:
            try:
                for m, a in zip(self.moteurs, cinematique_inverse(self.g, cible_x, cible_y)):
                    m.demander(a)
            except ValueError:
                self.poses_hors_portee += 1
        else:
            self.data.ctrl[:] = (cible_x, cible_y)

    def inclinaison_reelle(self):
        tx, ty = inclinaison_plateau(self.data)
        return tx - self.p.biais_x, ty - self.p.biais_y

    def angles_moteurs(self):
        return tuple(self.data.joint(f"moteur{i + 1}").qpos[0] for i in range(3)) if self.mecanisme else ()

    def couples(self):
        return tuple(self.data.actuator_force) if self.mecanisme else ()

    def lire_sans_contact_periode(self) -> int:
        n, self.pas_sans_contact_periode = self.pas_sans_contact_periode, 0
        return n

    def pousser(self, dvx: float, dvy: float) -> None:
        """Choc : ajoute (Δvx, Δvy) à la vitesse de la balle, avec la rotation de roulement associée."""
        adr = self.model.jnt_dofadr[self.model.body("balle").jntadr[0]]
        self.data.qvel[adr:adr + 2] += (dvx, dvy)
        # La vitesse de rotation d'une liberté (freejoint) s'exprime dans le repère de la balle
        R = self.data.body("balle").xmat.reshape(3, 3)
        r = self.p.rayon_balle
        self.data.qvel[adr + 3:adr + 6] += R.T @ np.array([-dvy / r, dvx / r, 0.0])

    def pas(self, dt: float) -> None:
        if self.mecanisme:
            for i, m in enumerate(self.moteurs):
                self.data.ctrl[i] = m.avancer(dt)
        mujoco.mj_step(self.model, self.data)
        if self.data.ncon == 0:           # seul le contact balle-plateau est activé dans le modèle
            self.pas_sans_contact += 1
            self.pas_sans_contact_periode += 1


# --- Validation de la physique ---------------------------------------------------------------------
def acceleration_roulement(k: float, theta: float, solref: str = "0.005 1", duree: float = 1.0,
                           frottement: float = 1.0) -> float:
    """Accélération mesurée par MuJoCo d'une balle (J = k·m·r²) sur un plan fixe incliné de θ.

    Théorie : ẍ = g·sin θ / (1 + k) si elle roule, g·(sin θ - μ·cos θ) si elle glisse.
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


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for nom, mecanisme in (("plateau.xml", False), ("robot.xml", True)):
        chemin = os.path.join("results", nom)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(xml_modele(Config(mecanisme=mecanisme, depart=(0.0, 0.0))))
        print(f"Modèle écrit : {chemin}")
