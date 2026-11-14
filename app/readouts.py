"""What the panel shows of the robot, read from a session: the true values of the simulation, not those measured by
the sensor of the robot. Without Qt."""
import math

from simulation.mujoco_physics import plate_tilt


def readouts(session) -> dict:
    """Current values; None for those of the ball once it has fallen."""
    phys = session.phys
    tx, ty = plate_tilt(phys.data)                 # true tilt of the plate, calibration error included
    torques = phys.torques()
    values = dict(tilt_x_deg=math.degrees(tx), tilt_y_deg=math.degrees(ty),
                  torque_Nm=max(abs(c) for c in torques) if torques else None,
                  distance_mm=None, speed_mm_s=None, height_mm=None, x_mm=None, y_mm=None)
    if session.state != "fallen":
        bx, by = phys.ball()
        xr, yr, _, _ = session.sim.cfg.trajectory(session.t)
        vx, vy = phys.ball_velocity()
        values.update(distance_mm=1000 * math.hypot(bx - xr, by - yr), speed_mm_s=1000 * math.hypot(vx, vy),
                      height_mm=1000 * max(0.0, phys.ball_height()), x_mm=1000 * bx, y_mm=1000 * by)
    return values


def distance_history(session) -> list:
    """Distance of the ball to the setpoint (mm) at every control period of the last 10 s (session.curves)."""
    return [1000 * c[1] for c in session.curves]
