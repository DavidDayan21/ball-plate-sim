"""Physique MuJoCo : modèles 3D (générés depuis la configuration) et interface de simulation.

Deux modèles, avec une caméra fixe « camera_haut » au-dessus du plateau :
    - mecanisme=False : plateau sur cardan (2 articulations), angles appliqués par deux servos
    - mecanisme=True  : robot complet, 3 × (moteur -> bras -> coude -> biellette), plateau libre ;
      chaque biellette est reliée au plateau par une contrainte « connect » (rotule), qui ferme
      la chaîne : 6 + 3 + 3 - 3×3 = 3 degrés de liberté, pilotés par les 3 moteurs.

Exporter les modèles pour les ouvrir dans le visualiseur MuJoCo :
    python -m simulation.mujoco_physics
    python -m mujoco.viewer --mjcf=results/robot.xml
"""
import math
import os

import mujoco
import numpy as np

from controller.kinematics import inverse_kinematics
from simulation.config import Config, MujocoParams
from simulation.motor import Motor

# --- Contact balle-plateau ------------------------------------------------------------------------
AIRBORNE_HEIGHT = 1e-4      # (m) balle « décollée » : plus de 0,1 mm au-dessus du plateau. En dessous, les
                        # micro-séparations (quelques µm) viennent de la souplesse numérique du contact


def contact_solref(pm: MujocoParams) -> str:
    """Contact ressort-amortisseur réglé directement (valeurs négatives de solref : -k et -b).

    Accélération de rappel : a = -k·p - b·v, avec k = ω², b = 2·ζ·ω, ω = 2π·f.
    Pour un oscillateur amorti, la restitution vaut e = exp(-π·ζ/√(1-ζ²)), donc
    ζ = -ln e / √(π² + ln² e). Mis sur les deux objets : MuJoCo n'a alors rien à mélanger.
    """
    w = 2 * math.pi * pm.contact_frequency
    ln_e = math.log(pm.restitution)
    zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
    return f"{-w * w:.1f} {-2 * zeta * w:.4f}"


# --- Génération des modèles (format MJCF) ------------------------------------------------------
HEADER = """<mujoco model="{name}">
  <option timestep="{dt}" integrator="implicitfast"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <visual><global azimuth="120" elevation="-30"/></visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.04" rgba="0.85 0.85 0.85 1"/>
    <camera name="top_camera" pos="0 0 {z_camera}" xyaxes="1 0 0 0 1 0" fovy="{fovy}"/>"""


def _ball(cfg: Config) -> str:
    pp, pm = cfg.plate, cfg.mujoco_params
    r = pp.ball_radius
    k = pp.k_inertia if pm.k_actual is None else pm.k_actual
    inertia = k * pm.ball_mass * r * r                 # J = k·m·r², donnée explicitement
    x0, y0 = cfg.start_position
    return f"""
    <body name="ball" pos="{x0} {y0} {cfg.geometry.height + r}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{r}" contype="1" conaffinity="1"
            solref="{contact_solref(pm)}" friction="{pm.friction} 0.005 0.0001" rgba="1 0.5 0 1"/>
      <inertial pos="0 0 0" mass="{pm.ball_mass}" diaginertia="{inertia} {inertia} {inertia}"/>
    </body>"""


def _plate_geom(cfg: Config) -> str:
    pp, pm = cfg.plate, cfg.mujoco_params
    e = pm.plate_thickness
    return (f'<geom name="plate" type="cylinder" size="{pp.plate_radius} {e / 2}" pos="0 0 {-e / 2}" '
            f'mass="{pm.plate_mass}" contype="1" conaffinity="1" solref="{contact_solref(pm)}" '
            f'friction="{pm.friction} 0.005 0.0001" '
            f'rgba="0.7 0.85 1 0.5"/>')


def model_xml(cfg: Config) -> str:
    g, pm = cfg.geometry, cfg.mujoco_params
    h = g.height
    header = HEADER.format(name="balancing robot" if cfg.mechanism else "angle-driven plate",
                           dt=cfg.dt_physics, z_camera=h + cfg.camera.height, fovy=cfg.camera.fovy)
    if not cfg.mechanism:
        # Articulations : θx (axe y), puis θy (axe -x) : rotation R = Ry(θx)·Rx(-θy)
        return header + f"""
    <geom name="stand" type="cylinder" size="0.01 {h / 2 + 0.02}" pos="0 0 {h / 2 - 0.02}" rgba="0.3 0.3 0.3 1"/>
    <body name="gimbal" pos="0 0 {h}">
      <joint name="tilt_x" type="hinge" axis="0 1 0" range="-30 30"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-7 1e-7 1e-7"/>
      <body name="plate">
        <joint name="tilt_y" type="hinge" axis="-1 0 0" range="-30 30"/>
        {_plate_geom(cfg)}
      </body>
    </body>{_ball(cfg)}
  </worldbody>
  <actuator>
    <position name="servo_x" joint="tilt_x" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
    <position name="servo_y" joint="tilt_y" kp="{pm.kp_servo}" kv="{pm.kv_servo}"/>
  </actuator>
</mujoco>
"""
    l1, l2 = g.arm_length, g.rod_length
    legs, constraints, actuators = "", "", ""
    for i in range(3):
        ux, uy = g.direction(i)
        bx, by = g.base_radius * ux, g.base_radius * uy
        axis = f"{uy} {-ux} 0"            # u × z : un angle α > 0 lève le bras
        legs += f"""
    <geom name="mount{i + 1}" type="box" size="0.012 0.012 0.02" pos="{bx} {by} -0.02" rgba="0.2 0.2 0.2 1"/>
    <body name="arm{i + 1}" pos="{bx} {by} 0">
      <joint name="motor{i + 1}" type="hinge" axis="{axis}" range="-60 60" armature="{pm.armature}"/>
      <geom type="capsule" fromto="0 0 0 {l1 * ux} {l1 * uy} 0" size="0.006" mass="{pm.arm_mass}" rgba="0.2 0.4 0.8 1"/>
      <body name="rod{i + 1}" pos="{l1 * ux} {l1 * uy} 0">
        <joint name="elbow{i + 1}" type="hinge" axis="{axis}" damping="1e-4"/>
        <geom type="capsule" fromto="0 0 0 0 0 {l2}" size="0.004" mass="{pm.rod_mass}" rgba="0.55 0.55 0.6 1"/>
      </body>
    </body>"""
        constraints += f'\n    <connect body1="rod{i + 1}" body2="plate" anchor="0 0 {l2}" solref="0.005 1"/>'
        actuators += (f'\n    <position name="servo{i + 1}" joint="motor{i + 1}" kp="{pm.kp_servo}" '
                        f'kv="{pm.kv_servo}" forcerange="{-pm.max_torque} {pm.max_torque}"/>')
    return header + legs + f"""
    <body name="plate" pos="0 0 {h}">
      <freejoint/>
      {_plate_geom(cfg)}
    </body>{_ball(cfg)}
  </worldbody>
  <equality>{constraints}
  </equality>
  <actuator>{actuators}
  </actuator>
</mujoco>
"""


def build(cfg: Config):
    """Modèle + données MuJoCo, balle placée avec sa vitesse initiale."""
    model = mujoco.MjModel.from_xml_string(model_xml(cfg))
    data = mujoco.MjData(model)
    # Vitesse initiale avec la rotation du roulement sans glissement : v + ω × (-r·ez) = 0
    vx, vy = cfg.start_velocity
    r = cfg.plate.ball_radius
    adr = model.jnt_dofadr[model.body("ball").jntadr[0]]
    data.qvel[adr:adr + 6] = (vx, vy, 0.0, -vy / r, vx / r, 0.0)
    mujoco.mj_forward(model, data)        # calcule les positions initiales (sinon xpos vaut 0)
    return model, data


def plate_tilt(data):
    """(θx, θy) du plateau, d'après sa normale n = R·ez = (sin θx cos θy, sin θy, cos θx cos θy)."""
    n = data.body("plate").xmat.reshape(3, 3)[:, 2]
    return math.atan2(n[0], n[2]), math.asin(max(-1.0, min(1.0, n[1])))


# --- Interface de simulation (même interface que PhysiqueSimple) -----------------------------------
class MujocoPhysics:
    def __init__(self, cfg: Config):
        self.cfg, self.p, self.g = cfg, cfg.plate, cfg.geometry
        self.mechanism = cfg.mechanism
        self.model, self.data = build(cfg)
        self.motors = ([Motor(0.0, cfg.max_speed, cfg.resolution, max_acceleration=cfg.max_acceleration)
                         for _ in range(3)] if self.mechanism else [])
        self.unreachable_poses = 0
        self.airborne_steps = 0                 # pas où la balle est décollée, au total
        self.airborne_steps_period = 0         # depuis la dernière lecture

    def ball_3d(self):
        """Centre de la balle dans le repère du plateau (x, y, z)."""
        plate = self.data.body("plate")
        R = plate.xmat.reshape(3, 3)
        return R.T @ (self.data.body("ball").xpos - plate.xpos)

    def ball(self):
        bx, by, _ = self.ball_3d()
        return float(bx), float(by)

    def fallen(self) -> bool:
        bx, by, bz = self.ball_3d()
        return math.hypot(bx, by) > self.p.plate_radius or bz < 0

    def command(self, theta_x: float, theta_y: float) -> None:
        # Défaut d'étalonnage : l'angle réellement visé = angle commandé + biais (inconnu du PID)
        target_x, target_y = theta_x + self.p.bias_x, theta_y + self.p.bias_y
        if self.mechanism:
            try:
                for m, a in zip(self.motors, inverse_kinematics(self.g, target_x, target_y)):
                    m.request(a)
            except ValueError:
                self.unreachable_poses += 1
        else:
            self.data.ctrl[:] = (target_x, target_y)

    def actual_tilt(self):
        tx, ty = plate_tilt(self.data)
        return tx - self.p.bias_x, ty - self.p.bias_y

    def motor_angles(self):
        return tuple(self.data.joint(f"motor{i + 1}").qpos[0] for i in range(3)) if self.mechanism else ()

    def torques(self):
        return tuple(self.data.actuator_force) if self.mechanism else ()

    def read_period_airborne_steps(self) -> int:
        n, self.airborne_steps_period = self.airborne_steps_period, 0
        return n

    def _ball_address(self):
        jnt = self.model.body("ball").jntadr[0]
        return self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]

    def ball_velocity(self):
        """(vx, vy) du centre de la balle dans le repère du plateau (m/s)."""
        _, dof = self._ball_address()
        R = self.data.body("plate").xmat.reshape(3, 3)
        v = R.T @ self.data.qvel[dof:dof + 3]   # vitesse linéaire d'une liberté : repère du monde
        return float(v[0]), float(v[1])

    def ball_height(self) -> float:
        """Hauteur du bas de la balle au-dessus de la surface du plateau (m) ; < 0 : enfoncement."""
        return float(self.ball_3d()[2]) - self.p.ball_radius

    def place_ball(self, x, y, z=0.0, vx=0.0, vy=0.0, vz=0.0) -> None:
        """Téléporte la balle : position (x, y) et hauteur z au-dessus du plateau, vitesse (vx, vy, vz),
        le tout dans le repère du plateau. La rotation est celle du roulement sans glissement."""
        qadr, dof = self._ball_address()
        plate = self.data.body("plate")
        R = plate.xmat.reshape(3, 3)
        r = self.p.ball_radius
        self.data.qpos[qadr:qadr + 3] = plate.xpos + R @ np.array([x, y, r + z])
        self.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)     # orientation neutre
        self.data.qvel[dof:dof + 3] = R @ np.array([vx, vy, vz])
        self.data.qvel[dof + 3:dof + 6] = R @ np.array([-vy / r, vx / r, 0.0])  # repère balle = monde
        mujoco.mj_forward(self.model, self.data)

    def push(self, dvx: float, dvy: float) -> None:
        """Choc : ajoute (Δvx, Δvy) à la vitesse de la balle, avec la rotation de roulement associée."""
        adr = self.model.jnt_dofadr[self.model.body("ball").jntadr[0]]
        self.data.qvel[adr:adr + 2] += (dvx, dvy)
        # La vitesse de rotation d'une liberté (freejoint) s'exprime dans le repère de la balle
        R = self.data.body("ball").xmat.reshape(3, 3)
        r = self.p.ball_radius
        self.data.qvel[adr + 3:adr + 6] += R.T @ np.array([-dvy / r, dvx / r, 0.0])

    def step(self, dt: float) -> None:
        if self.mechanism:
            for i, m in enumerate(self.motors):
                self.data.ctrl[i] = m.advance(dt)
        mujoco.mj_step(self.model, self.data)
        # Seul le contact balle-plateau est activé : sans contact, on mesure la hauteur de la balle
        if self.data.ncon == 0 and self.ball_height() > AIRBORNE_HEIGHT:
            self.airborne_steps += 1
            self.airborne_steps_period += 1


# --- Validation de la physique ---------------------------------------------------------------------
def _plane_and_ball(theta: float, k: float, pm: MujocoParams, friction: float, dt: float, z0: float = 0.0):
    """Plan fixe incliné de θ + balle (J = k·m·r²) posée dessus, ou lâchée de z0 au-dessus."""
    r, m = 0.02, pm.ball_mass
    inertia = k * m * r * r
    contact = f'solref="{contact_solref(pm)}" friction="{friction} 0.005 0.0001"'
    xml = f"""<mujoco><option timestep="{dt}" integrator="implicitfast"/><worldbody>
      <body euler="0 {math.degrees(theta)} 0"><geom type="box" size="0.5 0.5 0.005" pos="0 0 -0.005" {contact}/></body>
      <body name="ball"><freejoint/><geom type="sphere" size="{r}" {contact}/>
        <inertial pos="0 0 0" mass="{m}" diaginertia="{inertia} {inertia} {inertia}"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = np.array([math.sin(theta), 0.0, math.cos(theta)])        # normale au plan
    data.qpos[:3] = n * (r + z0)
    return model, data, r


def rolling_acceleration(k: float, theta: float, friction: float = 1.0, pm: MujocoParams = None,
                           dt: float = 0.0005, duration: float = 1.0) -> float:
    """Accélération mesurée par MuJoCo d'une balle posée sur un plan fixe incliné de θ.

    Théorie : ẍ = g·sin θ / (1 + k) si elle roule, g·(sin θ - μ·cos θ) si elle glisse.
    """
    model, data, _ = _plane_and_ball(theta, k, pm or MujocoParams(), friction, dt)
    tangent = np.array([math.cos(theta), 0.0, -math.sin(theta)])  # direction de la pente
    T, S = [], []
    while data.time < duration:
        mujoco.mj_step(model, data)
        T.append(data.time)
        S.append(float(data.qpos[:3] @ tangent))
    T, S = np.array(T), np.array(S)
    keep = T > 0.2                                              # on ignore la mise en contact
    return 2 * np.polyfit(T[keep], S[keep], 2)[0]               # s = ½·a·t² + ...


def measured_restitution(pm: MujocoParams = None, height: float = 0.305, dt: float = 0.0005) -> float:
    """Lâche la balle de `hauteur` sur un plan fixe horizontal : e = √(h_rebond / h_chute)."""
    model, data, r = _plane_and_ball(0.0, 2 / 3, pm or MujocoParams(), 1.0, dt, z0=height)
    z = []
    while data.time < 1.2:
        mujoco.mj_step(model, data)
        z.append(data.qpos[2] - r)
    z = np.array(z)
    impact = int(np.argmin(z[:int(0.5 / dt)]))
    return math.sqrt(max(0.0, z[impact:].max()) / height)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for name, mechanism in (("plate.xml", False), ("robot.xml", True)):
        path = os.path.join("results", name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(model_xml(Config(mechanism=mechanism, start_position=(0.0, 0.0))))
        print(f"Modèle écrit : {path}")
