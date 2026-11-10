"""MuJoCo physics: 3D models (generated from the configuration) and simulation interface.

Two models, with a fixed camera "top_camera" above the plate:
    - mechanism=False: plate on a gimbal (2 joints), angles applied by two servos; they follow their
      target with the profile of the motors (simulation/motor.py), without steps
    - mechanism=True:  complete robot, 3 × (motor -> arm -> elbow -> rod), free plate;
      each rod is connected to the plate by a "connect" constraint (ball joint), which closes
      the chain: 6 + 3 + 3 - 3×3 = 3 degrees of freedom, driven by the 3 motors.

Export the models to open them in the MuJoCo viewer:
    python -m simulation.mujoco_physics
    python -m mujoco.viewer --mjcf=results/robot.xml
"""
import math
import os

import mujoco
import numpy as np

from controller.kinematics import inverse_kinematics
from simulation.config import Config, MujocoParams
from simulation.motor import motors

# --- Ball-plate contact ---------------------------------------------------------------------------
AIRBORNE_HEIGHT = 1e-4      # (m) "airborne" ball: more than 0.1 mm above the plate. Below that, the
                            # micro-separations (a few µm) come from the numerical softness of the contact
SETTLING_TIME = 0.2         # (s) before the ball is placed, the robot reaches its command 0 and comes to rest
                            # (measured: within 0.01° in 0.15 s for the 3 arms, 0.1 s for the angle-driven plate)


def contact_solref(pm: MujocoParams) -> str:
    """Spring-damper contact set directly (negative solref values: -k and -b).

    Restoring acceleration: a = -k·p - b·v, with k = ω², b = 2·ζ·ω, ω = 2π·f.
    For a damped oscillator, the restitution is e = exp(-π·ζ/√(1-ζ²)), hence
    ζ = -ln e / √(π² + ln² e). Set on both objects: MuJoCo then has nothing to mix.
    """
    w = 2 * math.pi * pm.contact_frequency
    ln_e = math.log(pm.restitution)
    zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
    return f"{-w * w:.1f} {-2 * zeta * w:.4f}"


# --- Model generation (MJCF format) ------------------------------------------------------------
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
    inertia = k * pm.ball_mass * r * r                 # J = k·m·r², given explicitly
    x0, y0 = cfg.start_position
    return f"""
    <body name="ball" pos="{x0} {y0} {cfg.geometry.height + r + cfg.start_height}">
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
        # Joints: θx (y axis), then θy (-x axis): rotation R = Ry(θx)·Rx(-θy)
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
        axis = f"{uy} {-ux} 0"            # u × z: an angle α > 0 raises the arm
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
    """MuJoCo model + data: level plate, ball at rest at its start position."""
    model = mujoco.MjModel.from_xml_string(model_xml(cfg))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)        # computes the initial positions (otherwise xpos is 0)
    return model, data


def plate_tilt(data):
    """(θx, θy) of the plate, from its normal n = R·ez = (sin θx cos θy, sin θy, cos θx cos θy)."""
    n = data.body("plate").xmat.reshape(3, 3)[:, 2]
    return math.atan2(n[0], n[2]), math.asin(max(-1.0, min(1.0, n[1])))


# --- Simulation interface (same interface as SimplePhysics) ---------------------------------------
class MujocoPhysics:
    def __init__(self, cfg: Config):
        self.cfg, self.p, self.g = cfg, cfg.plate, cfg.geometry
        self.mechanism = cfg.mechanism
        self.model, self.data = build(cfg)
        self.motors = (motors(cfg, (0.0, 0.0, 0.0), cfg.resolution) if self.mechanism
                       else motors(cfg, (0.0, 0.0), 0.0))       # the 3 motors, or the 2 angles of the plate
        self.unreachable_poses = 0
        self.airborne_steps = 0                # steps with the ball airborne, in total
        self.airborne_steps_period = 0         # since the last read
        self.switch_on()

    def switch_on(self) -> None:
        """Start like a robot that is already on when the ball is placed: the plate first reaches the command 0,
        i.e. its calibration error, and comes to rest under its weight, with the ball held away; then the ball
        is placed at its start position, height and velocity, in the plate frame, and the time starts at 0.
        (Built level, the plate would otherwise move to its calibration error and sag under its weight during
        the first period, under the ball.)"""
        cfg = self.cfg
        qadr, dof = self._ball_address()
        held = (0.0, 0.0, cfg.geometry.height + 0.3, 1.0, 0.0, 0.0, 0.0)    # 30 cm above the plate: no contact
        self.command(0.0, 0.0)
        for _ in range(round(SETTLING_TIME / cfg.dt_physics)):
            self.data.qpos[qadr:qadr + 7] = held
            self.data.qvel[dof:dof + 6] = 0.0
            self.step(cfg.dt_physics)
        self.data.time = 0.0
        self.airborne_steps = self.airborne_steps_period = 0
        mujoco.mj_forward(self.model, self.data)
        (x0, y0), (vx, vy) = cfg.start_position, cfg.start_velocity
        self.place_ball(x0, y0, cfg.start_height, vx, vy)

    def ball_3d(self):
        """Center of the ball in the plate frame (x, y, z)."""
        plate = self.data.body("plate")
        R = plate.xmat.reshape(3, 3)
        return R.T @ (self.data.body("ball").xpos - plate.xpos)

    def ball(self):
        bx, by, _ = self.ball_3d()
        return float(bx), float(by)

    def fallen(self) -> bool:
        """Below the plate surface, or beyond its edge while level with it (center less than one diameter
        above the surface). A ball in the air beyond the edge of the tilted plate may still come back."""
        bx, by, bz = self.ball_3d()
        return bz < 0 or (math.hypot(bx, by) > self.p.plate_radius and bz < 2 * self.p.ball_radius)

    def command(self, theta_x: float, theta_y: float) -> None:
        # Calibration error: the angle actually targeted = commanded angle + bias (unknown to the PID)
        target_x, target_y = theta_x + self.p.bias_x, theta_y + self.p.bias_y
        if self.mechanism:
            try:
                for m, a in zip(self.motors, inverse_kinematics(self.g, target_x, target_y)):
                    m.request(a)
            except ValueError:
                self.unreachable_poses += 1
        else:
            for m, a in zip(self.motors, (target_x, target_y)):
                m.request(a)

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
        """(vx, vy) of the ball center in the plate frame (m/s)."""
        _, dof = self._ball_address()
        R = self.data.body("plate").xmat.reshape(3, 3)
        v = R.T @ self.data.qvel[dof:dof + 3]   # linear velocity of a free joint: world frame
        return float(v[0]), float(v[1])

    def ball_height(self) -> float:
        """Height of the bottom of the ball above the plate surface (m); < 0: penetration."""
        return float(self.ball_3d()[2]) - self.p.ball_radius

    def place_ball(self, x, y, z=0.0, vx=0.0, vy=0.0, vz=0.0) -> None:
        """Teleports the ball: position (x, y) and height z above the plate, velocity (vx, vy, vz),
        all in the plate frame. The rotation is the one of rolling without slipping."""
        qadr, dof = self._ball_address()
        plate = self.data.body("plate")
        R = plate.xmat.reshape(3, 3)
        r = self.p.ball_radius
        self.data.qpos[qadr:qadr + 3] = plate.xpos + R @ np.array([x, y, r + z])
        self.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)     # neutral orientation
        self.data.qvel[dof:dof + 3] = R @ np.array([vx, vy, vz])
        self.data.qvel[dof + 3:dof + 6] = R @ np.array([-vy / r, vx / r, 0.0])  # ball frame = world frame
        mujoco.mj_forward(self.model, self.data)

    def push(self, dvx: float, dvy: float) -> None:
        """Impulse: adds (Δvx, Δvy) to the ball velocity, with the matching rolling rotation."""
        adr = self.model.jnt_dofadr[self.model.body("ball").jntadr[0]]
        self.data.qvel[adr:adr + 2] += (dvx, dvy)
        # The angular velocity of a free joint is expressed in the ball frame
        R = self.data.body("ball").xmat.reshape(3, 3)
        r = self.p.ball_radius
        self.data.qvel[adr + 3:adr + 6] += R.T @ np.array([-dvy / r, dvx / r, 0.0])

    def step(self, dt: float) -> None:
        for i, m in enumerate(self.motors):
            self.data.ctrl[i] = m.advance(dt)
        mujoco.mj_step(self.model, self.data)
        # Only the ball-plate contact is enabled: with no contact, the height of the ball is checked
        if self.data.ncon == 0 and self.ball_height() > AIRBORNE_HEIGHT:
            self.airborne_steps += 1
            self.airborne_steps_period += 1


# --- Physics validation ----------------------------------------------------------------------------
def _plane_and_ball(theta: float, k: float, pm: MujocoParams, friction: float, dt: float, z0: float = 0.0):
    """Fixed plane tilted by θ + ball (J = k·m·r²) resting on it, or dropped from z0 above it."""
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
    n = np.array([math.sin(theta), 0.0, math.cos(theta)])        # normal to the plane
    data.qpos[:3] = n * (r + z0)
    return model, data, r


def rolling_acceleration(k: float, theta: float, friction: float = 1.0, pm: MujocoParams = None,
                           dt: float = 0.0005, duration: float = 1.0) -> float:
    """Acceleration measured by MuJoCo of a ball resting on a fixed plane tilted by θ.

    Theory: ẍ = g·sin θ / (1 + k) if it rolls, g·(sin θ - μ·cos θ) if it slides.
    """
    model, data, _ = _plane_and_ball(theta, k, pm or MujocoParams(), friction, dt)
    tangent = np.array([math.cos(theta), 0.0, -math.sin(theta)])  # direction of the slope
    T, S = [], []
    while data.time < duration:
        mujoco.mj_step(model, data)
        T.append(data.time)
        S.append(float(data.qpos[:3] @ tangent))
    T, S = np.array(T), np.array(S)
    keep = T > 0.2                                              # the initial contact phase is ignored
    return 2 * np.polyfit(T[keep], S[keep], 2)[0]               # s = ½·a·t² + ...


def measured_restitution(pm: MujocoParams = None, height: float = 0.305, dt: float = 0.0005) -> float:
    """Drops the ball from `height` onto a fixed horizontal plane: e = √(h_bounce / h_drop)."""
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
        print(f"Model written: {path}")
