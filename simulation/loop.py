"""Boucle fermée unique, quelle que soit la physique.

À chaque pas (0,5 ms) : la physique avance. Toutes les 20 ms (50 Hz) :
    mesure (capteur fictif ou caméra) -> contrôleur -> commande (θx, θy) -> moteurs / servos

    physique = "simple" : nos équations (simulation/simple_physics.py)
    physique = "mujoco" : moteur physique 3D (simulation/mujoco_physics.py)
"""
from dataclasses import dataclass, field
import math
import time

from controller.plate import PlateController
from simulation.sensor import Sensor
from simulation.config import G, Config


@dataclass
class Results:
    """Historique complet, échantillonné à chaque période de contrôle (données d'une tentative)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)                    # vraie position de la balle (m)
    y: list = field(default_factory=list)
    x_ref: list = field(default_factory=list)                # consigne (m)
    y_ref: list = field(default_factory=list)
    theta_x: list = field(default_factory=list)              # commandes du contrôleur (rad)
    theta_y: list = field(default_factory=list)
    actual_tilt: list = field(default_factory=list)   # (θx, θy) réels du plateau (rad)
    alphas: list = field(default_factory=list)               # angles des 3 moteurs (rad)
    velocity: list = field(default_factory=list)              # (vx, vy) de la balle, repère du plateau (m/s)
    height: list = field(default_factory=list)              # hauteur de la balle au-dessus du plateau (m)
    torques: list = field(default_factory=list)              # couples des 3 moteurs (N·m), MuJoCo
    airborne: list = field(default_factory=list)             # part de la période balle décollée, MuJoCo
    measurement_error: list = field(default_factory=list)        # |mesure - vraie position| avant latence (m)
    fallen: bool = False
    unreachable_poses: int = 0
    airborne_fraction: float = 0.0
    undetected: int = 0                                    # caméra : images sans balle détectée


def create_physics(cfg: Config):
    if cfg.physics == "simple":
        from simulation.simple_physics import SimplePhysics
        return SimplePhysics(cfg)
    if cfg.physics == "mujoco":
        from simulation.mujoco_physics import MujocoPhysics
        return MujocoPhysics(cfg)
    raise ValueError(f"Physique inconnue : {cfg.physics}")


class Simulation:
    """Simulation pas à pas (un pas = dt_physique).

    simuler() l'enchaîne sur toute la durée du scénario ; le bac à sable la pilote en direct.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.phys = create_physics(cfg)
        self.camera = None
        if cfg.sensor == "camera":
            self.create_camera()
        self.reset_control()
        self.push_steps = {round(tp / cfg.dt_physics): (dvx, dvy) for tp, dvx, dvy in cfg.pushes}
        self.steps_per_control = round(cfg.control_period / cfg.dt_physics)
        self.theta = (0.0, 0.0)             # dernière commande (connue du robot, sert à la caméra)
        self.last_measurement = cfg.start_position
        self.n = 0                          # numéro du prochain pas
        self.r = Results()

    def create_camera(self) -> None:
        """Caméra virtuelle. Son contexte OpenGL n'est utilisable que dans le fil (thread) qui l'a
        créée : la simulation doit tourner dans ce même fil."""
        from simulation.camera import BallCamera
        cfg = self.cfg
        self.camera = BallCamera(self.phys.model, cfg.camera, cfg.geometry.height,
                                  cfg.plate.ball_radius, cfg.tilt_correction)

    def reset_control(self) -> None:
        """(Re)crée le capteur et le contrôleur à partir de self.cfg, mémoire remise à zéro."""
        cfg = self.cfg
        if self.camera is not None:
            self.sensor = Sensor(sigma=0.0, latency=cfg.latency)   # le bruit vient de l'image
        else:
            self.sensor = Sensor(cfg.sigma, cfg.latency)
        self.ctrl = PlateController(cfg.kp, cfg.ki, cfg.kd(), cfg.theta_max, cfg.tau_d,
                                      model_gain=cfg.plate.coefficient * G, feedforward=cfg.feedforward,
                                      latency_compensation=cfg.latency if cfg.latency_compensation else 0,
                                      period=cfg.control_period)

    @property
    def t(self) -> float:
        return self.n * self.cfg.dt_physics    # pas de somme de dt : elle accumulerait des arrondis

    def step(self) -> bool:
        """Avance d'un pas. Renvoie False si la balle est tombée."""
        cfg, phys, r, n = self.cfg, self.phys, self.r, self.n
        t = self.t
        if phys.fallen():
            r.fallen = True
            return False

        if n % self.steps_per_control == 0:
            bx, by = phys.ball()
            if self.camera is not None:
                mx, my = self.camera.measure(phys.data, *self.theta, self.last_measurement)
                self.last_measurement = (mx, my)
            else:
                mx, my = bx, by
            xm, ym = self.sensor.measure(mx, my)
            if self.camera is None:         # capteur fictif : mesure bruitée de cet instant (avant latence)
                mx, my = self.sensor.queue[-1] if self.sensor.queue else (xm, ym)
            r.measurement_error.append(math.hypot(mx - bx, my - by))
            xr, yr, axr, ayr = cfg.trajectory(t)
            self.theta = self.ctrl.compute((xr, yr), (xm, ym), (axr, ayr), cfg.control_period)
            phys.command(*self.theta)
            for series, val in ((r.t, t), (r.x, bx), (r.y, by), (r.x_ref, xr), (r.y_ref, yr),
                               (r.theta_x, self.theta[0]), (r.theta_y, self.theta[1]),
                               (r.actual_tilt, phys.actual_tilt()),
                               (r.alphas, phys.motor_angles()),
                               (r.velocity, phys.ball_velocity()), (r.height, phys.ball_height())):
                series.append(val)
            if cfg.physics == "mujoco":
                r.torques.append(phys.torques())
                r.airborne.append(phys.read_period_airborne_steps() / self.steps_per_control)

        if n in self.push_steps:
            phys.push(*self.push_steps[n])
        phys.step(cfg.dt_physics)
        self.n += 1
        return True

    def finish(self) -> Results:
        """Libère les ressources et complète les indicateurs globaux."""
        r = self.r
        if self.camera is not None:
            r.undetected = self.camera.undetected
            self.camera.close()
        r.unreachable_poses = self.phys.unreachable_poses
        if self.cfg.physics == "mujoco":
            last = self.n if r.fallen else self.n - 1
            r.airborne_fraction = self.phys.airborne_steps / max(1, last)
        return r


def simulate(cfg: Config, view: bool = False) -> Results:
    """Simule le scénario de cfg. vue=True (MuJoCo) : fenêtre 3D en temps réel."""
    sim = Simulation(cfg)
    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.phys.model, sim.phys.data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.7, -30, 120
        viewer.cam.lookat[:] = (0, 0, cfg.geometry.height)
        start = time.perf_counter()

    for n in range(round(cfg.duration / cfg.dt_physics) + 1):
        if not sim.step():
            break
        if viewer is not None and n % sim.steps_per_control == 0:
            if not viewer.is_running():
                break
            viewer.sync()
            ahead = sim.phys.data.time - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)          # cale la simulation sur le temps réel

    if viewer is not None:
        viewer.close()
    return sim.finish()
