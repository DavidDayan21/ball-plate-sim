"""A simulated robot of the window: its session and its 3D view (app/view3d.py), and its fallen ball followed by the
camera of the view.

When the ball falls, the camera glides to look between the robot and the ball, and moves back to keep both in view
(the ball is followed up to FOLLOW_REACH from the center of the plate), for FALL_SHOW of simulated time; then the
ball comes back to the center of the plate, and the camera to where it was. A pause stops all of it.
"""
import math

FALL_SHOW = 2.0                 # (s) of simulated time showing the fallen ball
FOLLOW_REACH = 2.5              # (m) the camera follows the ball this far at most (a fast ball rolls about 2 m on
                                # the floor in 2 s)
FOLLOW_TIME = 0.3               # (s) time constant of the moves of the camera


class Robot:
    def __init__(self, session, view):
        self.session, self.view = session, view
        self.fall_time = None           # simulated time of the fall, while the ball is down
        self._home = None               # (lookat, distance) of the camera before the fall, until it is back there
        self._last_t = session.t

    @property
    def brain(self) -> str:
        return self.session.cfg.brain

    def home(self) -> tuple:
        """(lookat, distance) of the camera outside a fall."""
        camera = self.view.camera
        return self._home or (camera.lookat, camera.distance)

    def fall_left(self) -> float:
        """(s) Before the fallen ball comes back."""
        if self.fall_time is None:
            return FALL_SHOW
        return max(FALL_SHOW - (self.session.t - self.fall_time), 0.0)

    def follow_fall(self) -> None:
        """Once per frame. While the ball is down: the camera follows it, then the ball comes back to the center
        after FALL_SHOW. Afterwards, the camera goes back to where it was."""
        s, camera = self.session, self.view.camera
        dt, self._last_t = s.t - self._last_t, s.t
        if s.state == "fallen":
            if self.fall_time is None:
                self.fall_time, self._home = s.t, (camera.lookat, camera.distance)
            home_lookat, home_distance = self._home
            ball, plate = s.phys.data.body("ball").xpos, s.phys.data.body("plate").xpos
            dx, dy = ball[0] - plate[0], ball[1] - plate[1]
            reach = math.hypot(dx, dy)
            f = min(1.0, FOLLOW_REACH / reach) if reach > 0 else 0.0
            camera.glide((plate[0] + f * dx / 2, plate[1] + f * dy / 2, home_lookat[2]), home_distance + f * reach,
                         dt, FOLLOW_TIME)                   # halfway between the robot and the ball
            if s.t - self.fall_time >= FALL_SHOW:
                s.center_ball()
            return
        self.fall_time = None
        if self._home is not None:                          # back to the view of before the fall
            home_lookat, home_distance = self._home
            camera.glide(home_lookat, home_distance, dt, FOLLOW_TIME)
            if max(abs(a - b) for a, b in zip(camera.lookat, home_lookat)) < 1e-3:    # 1 mm: there
                camera.lookat, camera.distance, self._home = home_lookat, home_distance, None
