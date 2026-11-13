"""Single closed loop, whatever the physics and the brain.

At every step (0.5 ms): the physics advances. Every 20 ms (50 Hz):
    measurement (synthetic sensor or camera) -> brain -> command (θx, θy) -> motors / servos
The first measurement arrives one latency after the start: until then, the plate keeps its command.

    physics = "simple": our own equations (simulation/simple_physics.py)
    physics = "mujoco": 3D physics engine (simulation/mujoco_physics.py)

A brain is an object with:
    compute(setpoint, measurement, setpoint_acc, dt) -> (θx, θy) in rad, called at every control period;
    model_gain: the c·g of its model of the ball ((m/s²)/rad), nominal or estimated online; NaN if it has none.
cfg.brain chooses it: the PID brain (pid_brain, the default one) or the AI brain, a policy learned and saved in
brains/ (learned_brain). controller/policy.py holds the brain driven by a policy.
The motors, their ramps and the rule of the fast ramps (controller/ramps.py) are the same for every brain.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path
import time

from controller.adaptation import ModelGainEstimator
from controller.flight import FlightEstimator
from controller.network import Network
from controller.plate import PlateController
from controller.policy import OBSERVATION_SIZE, PolicyController
from controller.ramps import FastRamps
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
    model_gain: list = field(default_factory=list)           # c·g used by the brain ((m/s²)/rad): nominal, or
                                                             # estimated online (adapt_model_gain); NaN: no model
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


def gain_estimator(cfg: Config) -> ModelGainEstimator:
    """Estimator of the model gain c·g, from the value of the ball assumed by cfg."""
    return ModelGainEstimator(cfg.plate.coefficient * G, cfg.control_period, cfg.latency, cfg.theta_max)


def pid_brain(cfg: Config) -> PlateController:
    """The PID brain, with the settings of cfg."""
    flight = (FlightEstimator(cfg.geometry.height, cfg.plate.ball_radius, cfg.control_period, cfg.latency,
                              cfg.airborne_height, G) if cfg.airborne_height < math.inf else None)
    model_gain = cfg.plate.coefficient * G
    estimator = gain_estimator(cfg) if cfg.adapt_model_gain else None
    return PlateController(*cfg.pid_gains(), cfg.theta_max, cfg.tau_d,
                           model_gain=model_gain, feedforward=cfg.feedforward,
                           latency_compensation=cfg.latency if cfg.latency_compensation else 0,
                           period=cfg.control_period, integration_zone=cfg.integration_zone,
                           flight=flight, restitution=cfg.bounce_restitution, bounce_gain=cfg.bounce_gain,
                           wait_for_speed=cfg.wait_for_speed, estimator=estimator)


BRAINS = Path(__file__).resolve().parent.parent / "brains"       # the learned policies: <name>.npz


@lru_cache
def learned_policy(name: str) -> Network:
    """The policy brains/<name>.npz."""
    network = Network.load(BRAINS / f"{name}.npz")
    inputs = network.layers[0][0].shape[1]
    if inputs != OBSERVATION_SIZE:
        raise ValueError(f"The policy {name} was learned with an observation of {inputs} numbers: the AI brain now "
                         f"observes {OBSERVATION_SIZE} (controller/policy.py). It has to be learned again.")
    return network


def learned_brain(cfg: Config) -> PolicyController:
    """The AI brain: driven by the learned policy cfg.brain."""
    return PolicyController(learned_policy(cfg.brain), cfg.theta_max, cfg.latency, cfg.airborne_height,
                            gain_estimator(cfg))


def chosen_brain(cfg: Config):
    """The brain chosen by cfg.brain: "pid", the PID brain; otherwise the AI brain, driven by the learned policy
    brains/<cfg.brain>.npz."""
    return pid_brain(cfg) if cfg.brain == "pid" else learned_brain(cfg)


class Simulation:
    """Step-by-step simulation (one step = dt_physics).

    simulate() runs it over the whole scenario duration; the sandbox drives it live.
    brain: function cfg -> brain, called again at every reset_control(). None: the brain chosen by cfg.brain, read
    again at every reset_control() (the sandbox changes it).
    """

    def __init__(self, cfg: Config, brain=None):
        self.cfg = cfg
        self.create_brain = brain or chosen_brain
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
        """(Re)creates the sensor, the rule of the fast ramps and the brain from self.cfg, with their memory reset."""
        cfg = self.cfg
        if self.camera is not None:
            self.sensor = Sensor(sigma=0.0, latency=cfg.latency)   # the noise comes from the image
        else:
            self.sensor = Sensor(cfg.sigma, cfg.latency, cfg.seed, cfg.sigma_height)
        self.ramps = FastRamps(cfg.fast_speed)
        self.brain = self.create_brain(cfg)

    @property
    def t(self) -> float:
        return self.n * self.cfg.dt_physics    # no sum of dt: it would accumulate rounding errors

    def step(self) -> bool:
        """Advances one step. Returns False if the ball has fallen."""
        if self.phys.fallen():
            self.r.fallen = True
            return False
        inputs = self.sense()
        if inputs is not None:
            self.theta = self.brain.compute(*inputs, self.cfg.control_period)
        self.actuate()
        return True

    def sense(self):
        """First half of a step. At a control instant, measures the ball; returns what the brain receives,
        (setpoint, measurement, setpoint acceleration), or None when it has nothing to decide: between two control
        instants, and before the first image arrives (the plate keeps its command)."""
        cfg, phys, r = self.cfg, self.phys, self.r
        if self.n % self.steps_per_control != 0:
            return None
        t = self.t
        bx, by = phys.ball()
        if self.camera is not None:
            mx, my, mh = self.camera.measure(phys.data, *self.theta, self.last_measurement)
            self.last_measurement = (mx, my, mh)
        else:
            mx, my, mh = bx, by, phys.ball_height()
        measured = self.sensor.measure(mx, my, mh)
        if self.camera is None:             # synthetic sensor: noisy measurement of this instant (before latency)
            mx, my, mh = self.sensor.queue[-1] if self.sensor.queue else measured
        r.measurement_error.append(math.hypot(mx - bx, my - by))
        r.measured_height.append(mh)
        xr, yr, axr, ayr = cfg.trajectory(t)
        for series, val in ((r.t, t), (r.x, bx), (r.y, by), (r.x_ref, xr), (r.y_ref, yr)):
            series.append(val)
        if measured is None:
            return None
        self.ramps.update(measured, cfg.control_period)
        return (xr, yr), measured, (axr, ayr)

    def actuate(self) -> None:
        """Second half of a step. At a control instant, sends the command self.theta to the motors and records the
        period; then the physics advances by one step."""
        cfg, phys, r = self.cfg, self.phys, self.r
        if self.n % self.steps_per_control == 0:
            phys.command(*self.theta, fast=self.ramps.allowed)
            for series, val in ((r.theta_x, self.theta[0]), (r.theta_y, self.theta[1]),
                               (r.model_gain, self.brain.model_gain),
                               (r.actual_tilt, phys.actual_tilt()),
                               (r.alphas, phys.motor_angles()),
                               (r.velocity, phys.ball_velocity()), (r.height, phys.ball_height())):
                series.append(val)
            if cfg.physics == "mujoco":
                r.torques.append(phys.torques())
                r.airborne.append(phys.read_period_airborne_steps() / self.steps_per_control)
        if self.n in self.push_steps:
            phys.push(*self.push_steps[self.n])
        phys.step(cfg.dt_physics)
        self.n += 1

    def next_decision(self):
        """For a brain driven from outside (training): advances until it has to decide. Returns what it receives
        (see sense), or None if the ball has fallen. Its command is then written in self.theta, before actuate()."""
        while True:
            if self.phys.fallen():
                self.r.fallen = True
                return None
            inputs = self.sense()
            if inputs is not None:
                return inputs
            self.actuate()

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


def policy_brain(policy=None):
    """Brain driven by `policy` (controller/policy.py), as a function cfg -> brain for Simulation and simulate."""
    return lambda cfg: PolicyController(policy, cfg.theta_max, cfg.latency, cfg.airborne_height,
                                        gain_estimator(cfg))


def simulate(cfg: Config, view: bool = False, brain=None) -> Results:
    """Simulates the scenario of cfg. view=True (MuJoCo): real-time 3D window. brain: see Simulation."""
    sim = Simulation(cfg, brain)
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
