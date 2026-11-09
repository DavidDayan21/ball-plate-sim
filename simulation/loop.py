"""Single closed loop, whatever the physics.

At every step (0.5 ms): the physics advances. Every 20 ms (50 Hz):
    measurement (synthetic sensor or camera) -> controller -> command (θx, θy) -> motors / servos

    physics = "simple": our own equations (simulation/simple_physics.py)
    physics = "mujoco": 3D physics engine (simulation/mujoco_physics.py)
"""
from dataclasses import dataclass, field
import math
import time

from controller.plate import PlateController
from simulation.sensor import Sensor
from simulation.config import G, Config


@dataclass
class Results:
    """Full history, sampled at every control period (the data of one attempt)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)                    # true position of the ball (m)
    y: list = field(default_factory=list)
    x_ref: list = field(default_factory=list)                # setpoint (m)
    y_ref: list = field(default_factory=list)
    theta_x: list = field(default_factory=list)              # controller commands (rad)
    theta_y: list = field(default_factory=list)
    actual_tilt: list = field(default_factory=list)          # actual (θx, θy) of the plate (rad)
    alphas: list = field(default_factory=list)               # angles of the 3 motors (rad)
    velocity: list = field(default_factory=list)             # (vx, vy) of the ball, plate frame (m/s)
    height: list = field(default_factory=list)               # height of the ball above the plate (m)
    torques: list = field(default_factory=list)              # torques of the 3 motors (N·m), MuJoCo
    airborne: list = field(default_factory=list)             # fraction of the period with the ball airborne, MuJoCo
    measurement_error: list = field(default_factory=list)    # |measurement - true position| before latency (m)
    measured_height: list = field(default_factory=list)      # measured height of the ball, before latency (m)
    fallen: bool = False
    unreachable_poses: int = 0
    airborne_fraction: float = 0.0
    undetected: int = 0                                      # camera: images with no ball detected


def create_physics(cfg: Config):
    if cfg.physics == "simple":
        from simulation.simple_physics import SimplePhysics
        return SimplePhysics(cfg)
    if cfg.physics == "mujoco":
        from simulation.mujoco_physics import MujocoPhysics
        return MujocoPhysics(cfg)
    raise ValueError(f"Unknown physics: {cfg.physics}")


class Simulation:
    """Step-by-step simulation (one step = dt_physics).

    simulate() runs it over the whole scenario duration; the sandbox drives it live.
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
        self.theta = (0.0, 0.0)             # last command (known to the robot, used by the camera)
        self.last_measurement = (*cfg.start_position, cfg.start_height)
        self.n = 0                          # index of the next step
        self.r = Results()

    def create_camera(self) -> None:
        """Virtual camera. Its OpenGL context can only be used in the thread that
        created it: the simulation must run in that same thread."""
        from simulation.camera import BallCamera
        cfg = self.cfg
        self.camera = BallCamera(self.phys.model, cfg.camera, cfg.geometry.height,
                                  cfg.plate.ball_radius, cfg.tilt_correction, cfg.airborne_height)

    def reset_control(self) -> None:
        """(Re)creates the sensor and the controller from self.cfg, with their memory reset."""
        cfg = self.cfg
        if self.camera is not None:
            self.sensor = Sensor(sigma=0.0, latency=cfg.latency)   # the noise comes from the image
        else:
            self.sensor = Sensor(cfg.sigma, cfg.latency, cfg.seed, cfg.sigma_height)
        self.ctrl = PlateController(cfg.kp, cfg.ki, cfg.kd(), cfg.theta_max, cfg.tau_d,
                                      model_gain=cfg.plate.coefficient * G, feedforward=cfg.feedforward,
                                      latency_compensation=cfg.latency if cfg.latency_compensation else 0,
                                      period=cfg.control_period, integration_zone=cfg.integration_zone)

    @property
    def t(self) -> float:
        return self.n * self.cfg.dt_physics    # no sum of dt: it would accumulate rounding errors

    def step(self) -> bool:
        """Advances one step. Returns False if the ball has fallen."""
        cfg, phys, r, n = self.cfg, self.phys, self.r, self.n
        t = self.t
        if phys.fallen():
            r.fallen = True
            return False

        if n % self.steps_per_control == 0:
            bx, by = phys.ball()
            if self.camera is not None:
                mx, my, mh = self.camera.measure(phys.data, *self.theta, self.last_measurement)
                self.last_measurement = (mx, my, mh)
            else:
                mx, my, mh = bx, by, phys.ball_height()
            xm, ym, _ = self.sensor.measure(mx, my, mh)
            if self.camera is None:         # synthetic sensor: noisy measurement of this instant (before latency)
                mx, my, mh = self.sensor.queue[-1] if self.sensor.queue else (xm, ym, mh)
            r.measurement_error.append(math.hypot(mx - bx, my - by))
            r.measured_height.append(mh)
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
        """Releases the resources and completes the global metrics."""
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
    """Simulates the scenario of cfg. view=True (MuJoCo): real-time 3D window."""
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
                time.sleep(ahead)          # keeps the simulation in step with real time

    if viewer is not None:
        viewer.close()
    return sim.finish()
