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
from simulation.config import Config, ParametresMujoco
from simulation.moteur import Moteur

# --- Contact balle-plateau ------------------------------------------------------------------------
DECOLLEMENT = 1e-4      # (m) balle « décollée » : plus de 0,1 mm au-dessus du plateau. En dessous, les
                        # micro-séparations (quelques µm) viennent de la souplesse numérique du contact


def solref_contact(pm: ParametresMujoco) -> str:
    """Contact ressort-amortisseur réglé directement (valeurs négatives de solref : -k et -b).

    Accélération de rappel : a = -k·p - b·v, avec k = ω², b = 2·ζ·ω, ω = 2π·f.
    Pour un oscillateur amorti, la restitution vaut e = exp(-π·ζ/√(1-ζ²)), donc
    ζ = -ln e / √(π² + ln² e). Mis sur les deux objets : MuJoCo n'a alors rien à mélanger.
    """
    w = 2 * math.pi * pm.frequence_contact
    ln_e = math.log(pm.restitution)
    zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
    return f"{-w * w:.1f} {-2 * zeta * w:.4f}"


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
            solref="{solref_contact(pm)}" friction="{pm.frottement} 0.005 0.0001" rgba="1 0.5 0 1"/>
      <inertial pos="0 0 0" mass="{pm.masse_balle}" diaginertia="{inertie} {inertie} {inertie}"/>
    </body>"""


def _plateau_geom(cfg: Config) -> str:
    pp, pm = cfg.plateau, cfg.mujoco_params
    e = pm.epaisseur_plateau
    return (f'<geom name="plateau" type="cylinder" size="{pp.rayon_plateau} {e / 2}" pos="0 0 {-e / 2}" '
            f'mass="{pm.masse_plateau}" contype="1" conaffinity="1" solref="{solref_contact(pm)}" '
            f'friction="{pm.frottement} 0.005 0.0001" '
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
        self.pas_decolles = 0                 # pas où la balle est décollée, au total
        self.pas_decolles_periode = 0         # depuis la dernière lecture

    def balle_3d(self):
        """Centre de la balle dans le repère du plateau (x, y, z)."""
        plateau = self.data.body("plateau")
        R = plateau.xmat.reshape(3, 3)
        return R.T @ (self.data.body("balle").xpos - plateau.xpos)

    def balle(self):
        bx, by, _ = self.balle_3d()
        return float(bx), float(by)

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

    def lire_decolles_periode(self) -> int:
        n, self.pas_decolles_periode = self.pas_decolles_periode, 0
        return n

    def _adresse_balle(self):
        jnt = self.model.body("balle").jntadr[0]
        return self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]

    def vitesse_balle(self):
        """(vx, vy) du centre de la balle dans le repère du plateau (m/s)."""
        _, dof = self._adresse_balle()
        R = self.data.body("plateau").xmat.reshape(3, 3)
        v = R.T @ self.data.qvel[dof:dof + 3]   # vitesse linéaire d'une liberté : repère du monde
        return float(v[0]), float(v[1])

    def hauteur_balle(self) -> float:
        """Hauteur du bas de la balle au-dessus de la surface du plateau (m) ; < 0 : enfoncement."""
        return float(self.balle_3d()[2]) - self.p.rayon_balle

    def placer_balle(self, x, y, z=0.0, vx=0.0, vy=0.0, vz=0.0) -> None:
        """Téléporte la balle : position (x, y) et hauteur z au-dessus du plateau, vitesse (vx, vy, vz),
        le tout dans le repère du plateau. La rotation est celle du roulement sans glissement."""
        qadr, dof = self._adresse_balle()
        plateau = self.data.body("plateau")
        R = plateau.xmat.reshape(3, 3)
        r = self.p.rayon_balle
        self.data.qpos[qadr:qadr + 3] = plateau.xpos + R @ np.array([x, y, r + z])
        self.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)     # orientation neutre
        self.data.qvel[dof:dof + 3] = R @ np.array([vx, vy, vz])
        self.data.qvel[dof + 3:dof + 6] = R @ np.array([-vy / r, vx / r, 0.0])  # repère balle = monde
        mujoco.mj_forward(self.model, self.data)

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
        # Seul le contact balle-plateau est activé : sans contact, on mesure la hauteur de la balle
        if self.data.ncon == 0 and self.hauteur_balle() > DECOLLEMENT:
            self.pas_decolles += 1
            self.pas_decolles_periode += 1


# --- Validation de la physique ---------------------------------------------------------------------
def _plan_et_balle(theta: float, k: float, pm: ParametresMujoco, frottement: float, dt: float, z0: float = 0.0):
    """Plan fixe incliné de θ + balle (J = k·m·r²) posée dessus, ou lâchée de z0 au-dessus."""
    r, m = 0.02, pm.masse_balle
    inertie = k * m * r * r
    contact = f'solref="{solref_contact(pm)}" friction="{frottement} 0.005 0.0001"'
    xml = f"""<mujoco><option timestep="{dt}" integrator="implicitfast"/><worldbody>
      <body euler="0 {math.degrees(theta)} 0"><geom type="box" size="0.5 0.5 0.005" pos="0 0 -0.005" {contact}/></body>
      <body name="balle"><freejoint/><geom type="sphere" size="{r}" {contact}/>
        <inertial pos="0 0 0" mass="{m}" diaginertia="{inertie} {inertie} {inertie}"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = np.array([math.sin(theta), 0.0, math.cos(theta)])        # normale au plan
    data.qpos[:3] = n * (r + z0)
    return model, data, r


def acceleration_roulement(k: float, theta: float, frottement: float = 1.0, pm: ParametresMujoco = None,
                           dt: float = 0.0005, duree: float = 1.0) -> float:
    """Accélération mesurée par MuJoCo d'une balle posée sur un plan fixe incliné de θ.

    Théorie : ẍ = g·sin θ / (1 + k) si elle roule, g·(sin θ - μ·cos θ) si elle glisse.
    """
    model, data, _ = _plan_et_balle(theta, k, pm or ParametresMujoco(), frottement, dt)
    tangente = np.array([math.cos(theta), 0.0, -math.sin(theta)])  # direction de la pente
    T, S = [], []
    while data.time < duree:
        mujoco.mj_step(model, data)
        T.append(data.time)
        S.append(float(data.qpos[:3] @ tangente))
    T, S = np.array(T), np.array(S)
    garde = T > 0.2                                              # on ignore la mise en contact
    return 2 * np.polyfit(T[garde], S[garde], 2)[0]               # s = ½·a·t² + ...


def restitution_mesuree(pm: ParametresMujoco = None, hauteur: float = 0.305, dt: float = 0.0005) -> float:
    """Lâche la balle de `hauteur` sur un plan fixe horizontal : e = √(h_rebond / h_chute)."""
    model, data, r = _plan_et_balle(0.0, 2 / 3, pm or ParametresMujoco(), 1.0, dt, z0=hauteur)
    z = []
    while data.time < 1.2:
        mujoco.mj_step(model, data)
        z.append(data.qpos[2] - r)
    z = np.array(z)
    impact = int(np.argmin(z[:int(0.5 / dt)]))
    return math.sqrt(max(0.0, z[impact:].max()) / hauteur)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for nom, mecanisme in (("plateau.xml", False), ("robot.xml", True)):
        chemin = os.path.join("results", nom)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(xml_modele(Config(mecanisme=mecanisme, depart=(0.0, 0.0))))
        print(f"Modèle écrit : {chemin}")
