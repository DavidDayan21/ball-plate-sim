"""Modèles MuJoCo (format MJCF) générés à partir des paramètres Python des phases 2 et 3.

Générer le XML depuis Python garantit que MuJoCo utilise exactement la même géométrie
que la cinématique de la phase 3.

Les deux modèles contiennent une caméra fixe « camera_haut » au-dessus du plateau (phase 5).

Deux modèles :
    - xml_plateau : plateau sur un cardan (2 articulations), commandé directement en angles (4a)
    - xml_robot   : mécanisme complet à 3 bras, chaîne fermée par des contraintes « connect » (4b)

Pour exporter les fichiers XML et les ouvrir dans le visualiseur MuJoCo :
    python -m sim_mujoco.modeles
    python -m mujoco.viewer --mjcf=sim_mujoco/robot.xml
"""
from dataclasses import dataclass
import os

from controller.cinematique import Geometrie
from sim_mujoco.camera import ParametresCamera
from sim_simple.plateau import ParametresPlateau


@dataclass(frozen=True)
class ParametresMujoco:
    """Paramètres physiques ajoutés par MuJoCo (hypothèses, à recaler sur le vrai matériel)."""
    dt: float = 0.001                  # pas de simulation (s)
    masse_balle: float = 0.0027        # balle de ping-pong (kg)
    masse_plateau: float = 0.25        # disque acrylique Ø30 cm, 3 mm d'épaisseur (kg, estimation)
    epaisseur_plateau: float = 0.003   # (m)
    masse_bras: float = 0.02           # (kg)
    masse_biellette: float = 0.01      # (kg)
    kp_servo: float = 50.0             # raideur de l'asservissement de position (N·m/rad)
    kv_servo: float = 0.5              # amortissement de l'asservissement (N·m·s/rad)
    couple_max: float = 0.4            # couple max d'un moteur (N·m, ordre de grandeur d'un NEMA 17)
    armature: float = 5e-6             # inertie du rotor du moteur (kg·m²)
    solref_contact: str = "0.005 1"    # contact balle-plateau plus rigide que par défaut (voir le cours)


def _balle(pp: ParametresPlateau, pm: ParametresMujoco, x0: float, y0: float, hauteur: float) -> str:
    r = pp.rayon_balle
    inertie = pp.k_inertie * pm.masse_balle * r * r     # J = k·m·r², donnée explicitement
    return f"""
    <body name="balle" pos="{x0} {y0} {hauteur + r}">
      <freejoint/>
      <geom name="balle" type="sphere" size="{r}" contype="1" conaffinity="1"
            solref="{pm.solref_contact}" rgba="1 0.5 0 1"/>
      <inertial pos="0 0 0" mass="{pm.masse_balle}" diaginertia="{inertie} {inertie} {inertie}"/>
    </body>"""


def _plateau_geom(pp: ParametresPlateau, pm: ParametresMujoco) -> str:
    e = pm.epaisseur_plateau
    return (f'<geom name="plateau" type="cylinder" size="{pp.rayon_plateau} {e / 2}" pos="0 0 {-e / 2}" '
            f'mass="{pm.masse_plateau}" contype="1" conaffinity="1" rgba="0.7 0.85 1 0.5"/>')


ENTETE = """<mujoco model="{nom}">
  <option timestep="{dt}" integrator="implicitfast"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <visual><global azimuth="120" elevation="-30"/></visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="sol" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.04" rgba="0.85 0.85 0.85 1"/>
    <camera name="camera_haut" pos="0 0 {z_camera}" xyaxes="1 0 0 0 1 0" fovy="{fovy}"/>"""


def xml_plateau(pp: ParametresPlateau = None, pm: ParametresMujoco = None,
                depart=(0.0, 0.0), hauteur: float = 0.12, cam: ParametresCamera = None) -> str:
    """Plateau sur cardan au-dessus d'un pied. Articulations : θx (axe y), puis θy (axe -x).

    L'ordre des articulations reproduit la rotation R = Ry(θx)·Rx(-θy) de la phase 3.
    """
    pp, pm, cam = pp or ParametresPlateau(), pm or ParametresMujoco(), cam or ParametresCamera()
    return ENTETE.format(nom="plateau en angles", dt=pm.dt, z_camera=hauteur + cam.hauteur, fovy=cam.fovy) + f"""
    <geom name="pied" type="cylinder" size="0.01 {hauteur / 2 + 0.02}" pos="0 0 {hauteur / 2 - 0.02}" rgba="0.3 0.3 0.3 1"/>
    <body name="cardan" pos="0 0 {hauteur}">
      <joint name="inclinaison_x" type="hinge" axis="0 1 0" range="-30 30"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
      <body name="plateau">
        <joint name="inclinaison_y" type="hinge" axis="-1 0 0" range="-30 30"/>
        {_plateau_geom(pp, pm)}
      </body>
    </body>{_balle(pp, pm, depart[0], depart[1], hauteur)}
  </worldbody>
  <actuator>
    <position name="servo_x" joint="inclinaison_x" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
    <position name="servo_y" joint="inclinaison_y" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
  </actuator>
</mujoco>
"""


def xml_robot(g: Geometrie = None, pp: ParametresPlateau = None, pm: ParametresMujoco = None,
              depart=(0.0, 0.0), cam: ParametresCamera = None) -> str:
    """Mécanisme complet : 3 × (moteur -> bras -> coude -> biellette), plateau libre (6 DDL).

    Le coude est une articulation parallèle à l'axe moteur : chaque biellette reste dans son
    plan vertical radial. Le bout de chaque biellette est relié au plateau par une contrainte
    « connect » (rotule) : c'est elle qui ferme la chaîne cinématique.
    Bilan : 6 (plateau) + 3 (moteurs) + 3 (coudes) - 3×3 (rotules) = 3 degrés de liberté,
    pilotés par les 3 moteurs.
    """
    g, pp, pm = g or Geometrie(), pp or ParametresPlateau(), pm or ParametresMujoco()
    cam = cam or ParametresCamera()
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
    entete = ENTETE.format(nom="robot equilibriste", dt=pm.dt, z_camera=g.hauteur + cam.hauteur, fovy=cam.fovy)
    return entete + jambes + f"""
    <body name="plateau" pos="0 0 {g.hauteur}">
      <freejoint/>
      {_plateau_geom(pp, pm)}
    </body>{_balle(pp, pm, depart[0], depart[1], g.hauteur)}
  </worldbody>
  <equality>{contraintes}
  </equality>
  <actuator>{actionneurs}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    dossier = os.path.dirname(__file__)
    for nom, xml in (("plateau.xml", xml_plateau()), ("robot.xml", xml_robot())):
        chemin = os.path.join(dossier, nom)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(xml)
        print(f"Modèle écrit : {chemin}")
