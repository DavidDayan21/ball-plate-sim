"""The parts of the robot as shown in the window: a lab prototype built around the simulated mechanism. Without Qt.

Display only: meshes of the display scene (app/scene.py) that touch nothing (no contact) and that the physics never
sees. The moving parts are carried by the bodies of the physics, on its joints: each arm turns about the axis of its
motor and ends at the elbow, the lower rod end turns about the elbow, the rod is as long as the simulated one between
its joints, the plate and the ball are those of the physics. One liberty: the upper rod ends sit EYE_DEPTH below the
plate surface, where the physics puts the ball joints (no real part can: it would stick out of the plate). When a rod
leans on the plate by an angle a, the rod end and the bolt of its bracket are then 2·EYE_DEPTH·sin(a/2) apart: 1.8 mm
at 10° (measured over four hard catches: a up to 15.6°, 2.9 mm). The picture also shows the give of the simulated ball
joints as it is: 0.1 mm at rest, up to 1.9 mm in a hard catch (measured).

What is shown (leg i: u radial, m toward its motor, z up; the motor axis at base_radius·u, z = 0):
- an aluminium base on rubber pads, low enough for an arm down by 30° (measured over hard catches: 28°, the lowest
  moving part 1.7 mm above the base);
- per leg, a NEMA 17 stepper motor on a zinc-plated L bracket, a machined arm clamped on its shaft, a threaded rod
  between two rod ends, a bracket under the plate; the motor cables run on a ring around the camera, out of reach of
  the arms;
- the camera under the transparent plate, looking up, with a ring of LEDs, on brass standoffs;
- a small computer (the camera ribbon goes to it), a board with three stepper drivers, a LiPo battery: no cable leaves;
- a front panel: the name of the robot, two LEDs (PID, AI) with their engraved labels, the one of the brain lit;
- the seam of the ball and its printing, one per ball (it shows the ball turning, sliding or rolling).
Parts that depend on a choice (the printing of the ball, the label of the AI in the language of the interface) exist
once per value; app/scene.DisplayScene.show hides the others.
"""
from functools import lru_cache
import math

import numpy as np

from app import NAME
from app.i18n import DEFAULT, LANGUAGES, load
from app.meshes import (Frame, Mesh, box, chamfered_square, cylinder, disc, hexagon, lathe, prism, ribbon,
                        rounded_rectangle, smooth_path, stadium, text_flat, text_on_sphere, tube)
from simulation.session import BALLS

GROUND = -0.04                      # (m) the floor of the physics model (simulation/mujoco_physics.py)
BASE_R, BASE_TOP, BASE_BOTTOM = 0.13, -0.032, -0.037
NEMA = 0.02115                      # half side of a NEMA 17 motor
EYE_DEPTH = 0.0105                  # (m) centers of the upper rod ends, below the plate surface
CAMERA_Z = 0.024                    # board of the camera, bottom
LED_ON = (0.15, 1.0, 0.35)          # green
PRINTS = {"ping_pong": ("40+", "★★★"),                                  # the marks of a real ball
          "solid": ("2/5", "mr²"),                                               # its moment of inertia
          "heavy": (f"{1000 * BALLS['heavy']['ball_mass']:.0f} g",)}                  # its mass
DEFAULTS = {"ball": "ping_pong", "language": DEFAULT}

MATERIALS = {                       # name: (r, g, b), specular, shininess[, emission]
    "alu_base": ((0.72, 0.73, 0.75), 0.45, 0.35), "alu": ((0.80, 0.81, 0.83), 0.85, 0.75),
    "zinc": ((0.66, 0.67, 0.68), 0.7, 0.6), "steel": ((0.62, 0.63, 0.65), 1.0, 0.9),
    "chrome": ((0.88, 0.89, 0.90), 1.0, 1.0), "black": ((0.07, 0.07, 0.08), 0.45, 0.55),
    "oxide": ((0.16, 0.16, 0.17), 0.8, 0.8), "stack": ((0.55, 0.56, 0.58), 0.8, 0.7),
    "rubber": ((0.04, 0.04, 0.04), 0.05, 0.1), "pcb": ((0.05, 0.33, 0.17), 0.45, 0.6),
    "pcb_dark": ((0.06, 0.06, 0.07), 0.45, 0.6), "pcb_purple": ((0.30, 0.13, 0.42), 0.45, 0.6),
    "chip": ((0.09, 0.09, 0.10), 0.3, 0.4), "gold": ((0.86, 0.68, 0.30), 1.0, 0.9),
    "brass": ((0.78, 0.62, 0.30), 0.9, 0.8), "white_plastic": ((0.93, 0.92, 0.88), 0.2, 0.3),
    "cable": ((0.06, 0.06, 0.07), 0.35, 0.5), "cable_gray": ((0.55, 0.56, 0.58), 0.3, 0.4),
    "wire_red": ((0.75, 0.08, 0.06), 0.4, 0.5), "ribbon": ((0.85, 0.85, 0.82), 0.3, 0.4),
    "terminal": ((0.10, 0.45, 0.25), 0.3, 0.4), "capacitor": ((0.10, 0.16, 0.38), 0.6, 0.6),
    "lens": ((0.03, 0.05, 0.09), 1.0, 1.0), "led_white": ((1.0, 0.98, 0.92), 0.5, 0.5, 0.9),
    "led_status": ((0.2, 1.0, 0.4), 0.5, 0.5, 1.0), "lipo": ((0.12, 0.20, 0.45), 0.5, 0.6),
    "lipo_label": ((0.70, 0.71, 0.73), 0.6, 0.6), "strap": ((0.08, 0.08, 0.09), 0.05, 0.1),
    "xt60": ((0.95, 0.78, 0.10), 0.4, 0.5), "engraving": ((0.86, 0.86, 0.86), 0.1, 0.2),
    "seam": ((0.82, 0.36, 0.0), 0.5, 0.6), "print": ((0.10, 0.07, 0.05), 0.3, 0.4),
}


class Parts:
    """Meshes by (body, material, switch); switch: "" (always shown), "ball:<name>" or "language:<code>"."""

    def __init__(self):
        self.meshes = {}

    def __call__(self, body: str, material: str, mesh: Mesh, switch: str = "") -> None:
        self.meshes.setdefault((body, material, switch), []).append(mesh)

    def joined(self) -> dict:
        return {key: Mesh.join(meshes).oriented() for key, meshes in self.meshes.items()}


def socket_head(add, body, fr: Frame, r=0.0028, h=0.0025, z0=0.0):
    """Socket head cap screw along e3, its head from z0 to z0 + h."""
    add(body, "oxide", lathe(fr, [(0, z0), (r, z0), (r, z0 + h - 0.0003), (r - 0.0003, z0 + h), (0, z0 + h)], 24))
    add(body, "black", prism(fr, hexagon(0.0013), z0 + h, z0 + h + 0.00005))


def nut(add, body, fr: Frame, r_corner, z0, z1, material="zinc"):
    add(body, material, prism(fr, hexagon(r_corner, math.pi / 6), z0, z1))


def leg_axes(geometry, i):
    """(u, m, z) of leg i: radial, toward its motor (z × u), vertical."""
    ux, uy = geometry.direction(i)
    return np.array([ux, uy, 0.0]), np.array([-uy, ux, 0.0]), np.array([0.0, 0.0, 1.0])


@lru_cache(maxsize=4)
def robot_meshes(geometry, plate_radius: float, plate_thickness: float, ball_radius: float) -> dict:
    """{(body, material, switch): Mesh} of the robot of this geometry. Built once: 0.35 s on the development
    computer."""
    add = Parts()
    world = Frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    _base(add, world)
    for i in range(3):
        _leg(add, geometry, i)
    add("plate", "acrylic_edge", cylinder(world, plate_radius + 0.0003, -plate_thickness, 0.00002, 128,
                                          r_in=plate_radius - 0.0012))
    _camera(add, world)
    drivers = _electronics(add)
    _battery(add, drivers)
    _panel(add)
    _ball(add, ball_radius)
    return add.joined()


def _base(add, w):
    add("world", "alu_base", lathe(w, [(0, BASE_BOTTOM), (BASE_R - 0.0008, BASE_BOTTOM), (BASE_R, BASE_BOTTOM + 0.0008),
                                       (BASE_R, BASE_TOP - 0.0012), (BASE_R - 0.0012, BASE_TOP), (0, BASE_TOP)], 96))
    for k in range(4):
        a = math.radians(45 + 90 * k)
        add("world", "rubber", lathe(w.at(0.105 * math.cos(a), 0.105 * math.sin(a), 0),
                                     [(0, GROUND), (0.0105, GROUND), (0.0115, GROUND + 0.001), (0.0115, BASE_BOTTOM),
                                      (0, BASE_BOTTOM)], 32))


def _leg(add, g, i):
    u, m, z = leg_axes(g, i)
    b = g.base_radius * u
    L = Frame(b, u, m, z)                                      # fixed parts of the leg
    # L bracket of the motor (zinc-plated steel), its gussets and screws
    add("world", "zinc", box(L, -0.025, 0.025, 0.0115, 0.014, BASE_TOP, 0.023))
    add("world", "zinc", box(L, -0.025, 0.025, 0.014, 0.058, BASE_TOP, BASE_TOP + 0.0025))
    G = Frame(b, m, z, u)
    for s0, s1 in ((0.0225, 0.025), (-0.025, -0.0225)):
        add("world", "zinc", prism(G, [(0.014, BASE_TOP + 0.0025), (0.036, BASE_TOP + 0.0025), (0.014, 0.004)], s0, s1))
    toward_arm = Frame(b, z, u, -m)
    for zz, uu in ((0.0155, 0.0155), (0.0155, -0.0155), (-0.0155, 0.0155), (-0.0155, -0.0155)):
        socket_head(add, "world", toward_arm.at(zz, uu, 0), z0=-0.0115)
    add("world", "black", cylinder(toward_arm, 0.011, -0.0116, -0.0113, 48))      # boss of the motor, in the hole
    for uu in (-0.017, 0.017):
        socket_head(add, "world", Frame(b + uu * u + 0.0545 * m, u, m, z), z0=BASE_TOP + 0.0025, h=0.002)
    # NEMA 17 motor along m: end caps, laminated stack, screws, shaft, connector and its cable
    M = Frame(b, z, u, m)
    add("world", "black", prism(M, chamfered_square(NEMA, 0.0045), 0.014, 0.0215))
    add("world", "stack", prism(M, chamfered_square(NEMA - 0.0004, 0.0032), 0.0215, 0.0435))
    for k in range(7):
        s = 0.0225 + k * 0.0032
        add("world", "stack", prism(M, chamfered_square(NEMA, 0.0032), s, s + 0.0012))
    add("world", "black", prism(M, chamfered_square(NEMA, 0.0045), 0.0435, 0.0545))
    for zz, uu in ((0.0155, 0.0155), (0.0155, -0.0155), (-0.0155, 0.0155), (-0.0155, -0.0155)):
        add("world", "oxide", lathe(M.at(zz, uu, 0), [(0, 0.0545), (0.0027, 0.0545), (0.0027, 0.0565), (0, 0.0565)],
                                    20))
    add("world", "steel", cylinder(toward_arm, 0.0025, -0.0115, -0.0028, 24))
    add("world", "white_plastic", box(L, -NEMA - 0.0045, -NEMA, 0.0455, 0.0525, -0.0055, 0.0055))
    cable = [L.p(-NEMA - 0.0045, 0.049, 0), L.p(-0.0285, 0.049, -0.002), L.p(-0.0295, 0.046, -0.014),
             L.p(-0.0295, 0.036, BASE_TOP + 0.0019), L.p(-0.0295, 0.012, BASE_TOP + 0.0019)]
    add("world", "cable", tube(smooth_path(cable, 8), 0.0018, 12))
    # arm (body arm<i>): machined link from the motor axis to the elbow, clamping hub, bolt of the elbow
    arm, l1 = f"arm{i + 1}", g.arm_length
    add(arm, "alu", prism(Frame((0, 0, 0), u, z, -m), stadium(0.0105, 0.0068, l1), -0.008, -0.003, smooth=True))
    add(arm, "alu", cylinder(Frame((0, 0, 0), z, u, m), 0.0095, 0.008, 0.0109, 40))
    add(arm, "oxide", cylinder(Frame(0.00945 * m + 0.0089 * z, u, m, z), 0.0016, 0, 0.0010, 16))
    socket_head(add, arm, Frame(l1 * u, z, u, -m), z0=0.0025, h=0.0026)
    nut(add, arm, Frame(l1 * u, z, u, m), 0.0032, 0.008, 0.0104)
    add(arm, "steel", cylinder(Frame(l1 * u, z, u, m), 0.0015, 0.0104, 0.0114, 16))
    # rod (body rod<i>, origin at the elbow): two rod ends, lock nuts, threaded rod
    rod, top = f"rod{i + 1}", g.rod_length - EYE_DEPTH
    rf = math.sqrt(0.0042 ** 2 - 0.0034 ** 2)
    tilt = math.asin(0.0034 / 0.0042)
    for zc, sense in ((0.0, 1), (top, -1)):
        eye = Frame(zc * z, z, u, m)
        add(rod, "zinc", cylinder(eye, 0.0065, -0.0025, 0.0025, 40, r_in=0.0041))
        add(rod, "chrome", lathe(eye, [(0.0042 * math.cos(t), 0.0042 * math.sin(t))
                                       for t in np.linspace(-tilt, tilt, 10)], 32, smooth=True))
        add(rod, "chrome", disc(eye, rf, 0.0034, True, 0.0, 32))
        add(rod, "chrome", disc(eye, rf, -0.0034, False, 0.0, 32))
        R = Frame(zc * z, u, m, sense * z)
        add(rod, "zinc", lathe(R, [(0, 0.0058), (0.0036, 0.0058), (0.0036, 0.0150), (0.0031, 0.0155), (0, 0.0155)], 24))
        nut(add, rod, R, 0.0040, 0.0155, 0.0188)
    add(rod, "steel", cylinder(Frame((0, 0, 0), u, m, z), 0.002, 0.0188, top - 0.0188, 20))
    # under the plate (body plate): bracket of the upper rod end, its bolt, two screws through the plate
    a = (g.base_radius + g.arm_length) * u
    arc = np.linspace(math.pi, 2 * math.pi, 14)[1:-1]
    outline = ([(0.008, -0.003), (-0.008, -0.003), (-0.008, -EYE_DEPTH)]
               + [(0.008 * math.cos(t), -EYE_DEPTH + 0.008 * math.sin(t)) for t in arc] + [(0.008, -EYE_DEPTH)])
    add("plate", "alu", prism(Frame(a, u, z, -m), outline, -0.0078, -0.0028))
    add("plate", "alu", box(Frame(a, u, m, z), -0.008, 0.008, 0.0028, 0.016, -0.0055, -0.003))
    socket_head(add, "plate", Frame(a - EYE_DEPTH * z, z, u, -m), z0=0.0025, h=0.0026)
    nut(add, "plate", Frame(a - EYE_DEPTH * z, z, u, m), 0.0032, 0.0078, 0.0102)
    add("plate", "steel", cylinder(Frame(a - EYE_DEPTH * z, z, u, m), 0.0015, 0.0102, 0.0112, 16))
    for uu in (-0.0045, 0.0045):
        add("plate", "oxide", cylinder(Frame(a + uu * u + 0.0115 * m, u, m, z), 0.0025, -0.0002, 0.00015, 20))


def _camera(add, w):
    """Camera looking up through the plate, its ring of LEDs, on four brass standoffs; its ribbon to the computer."""
    for x in (-0.0105, 0.0105):
        for y in (-0.0105, 0.0105):
            add("world", "brass", prism(w.at(x, y, 0), hexagon(0.0026), BASE_TOP, CAMERA_Z))
            add("world", "zinc", cylinder(w.at(x, y, 0), 0.0019, CAMERA_Z + 0.001, CAMERA_Z + 0.0022, 16))
    add("world", "pcb", box(w, -0.0125, 0.0125, -0.012, 0.012, CAMERA_Z, CAMERA_Z + 0.001))
    top = CAMERA_Z + 0.001
    add("world", "black", box(w, -0.0043, 0.0043, -0.0043, 0.0043, top, top + 0.0045))
    add("world", "black", lathe(w, [(0, top + 0.0045), (0.0046, top + 0.0045), (0.0046, top + 0.0068),
                                    (0.0042, top + 0.0070), (0.0042, top + 0.0098), (0.0034, top + 0.0100),
                                    (0, top + 0.0100)], 32))
    add("world", "lens", disc(w, 0.0031, top + 0.01003, True, 0.0, 32))
    ring = top + 0.0062
    add("world", "pcb_dark", cylinder(w, 0.0155, ring, ring + 0.0010, 64, r_in=0.0075))
    for k in range(3):
        a = math.radians(90 + 120 * k)
        add("world", "brass", cylinder(w.at(0.0118 * math.cos(a), 0.0118 * math.sin(a), 0), 0.0009, top, ring, 10))
    for k in range(12):
        a = 2 * math.pi * (k + 0.5) / 12
        f = Frame((0.0115 * math.cos(a), 0.0115 * math.sin(a), 0), (math.cos(a), math.sin(a), 0),
                  (-math.sin(a), math.cos(a), 0), (0, 0, 1))
        add("world", "led_white", box(f, -0.0012, 0.0012, -0.0012, 0.0012, ring + 0.0010, ring + 0.0016))
    add("world", "chip", box(w, -0.004, 0.004, 0.0085, 0.012, top, top + 0.0012))
    path = [(0, 0.0120, top + 0.0004), (0, 0.0150, top - 0.0003), (0, 0.0172, top - 0.004), (0, 0.0182, 0.0),
            (0, 0.0190, BASE_TOP + 0.006), (0, 0.0225, BASE_TOP + 0.0006), (0, 0.040, BASE_TOP + 0.0004),
            (0, 0.048, BASE_TOP + 0.0004), (0, 0.056, BASE_TOP + 0.0025), (0, 0.0595, BASE_TOP + 0.0068),
            (0, 0.0650, BASE_TOP + 0.0059)]
    add("world", "ribbon", ribbon(smooth_path(path, 6), 0.0075, 0.0003))


def _board(add, center, angle, length, width, material, standoff=0.003, thickness=0.0012) -> Frame:
    """A circuit board on four brass standoffs, its length along `angle` (degrees). Its frame: top surface at 0."""
    a = math.radians(angle)
    f = Frame((center[0], center[1], BASE_TOP + standoff + thickness), (math.cos(a), math.sin(a), 0),
              (-math.sin(a), math.cos(a), 0), (0, 0, 1))
    add("world", material, box(f, -length / 2, length / 2, -width / 2, width / 2, -thickness, 0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = f.at(sx * (length / 2 - 0.0035), sy * (width / 2 - 0.0035), 0)
            add("world", "brass", prism(p, hexagon(0.0025), -thickness - standoff, -thickness))
            add("world", "zinc", cylinder(p, 0.0019, 0, 0.0012, 16))
    return f


def _ring(r, t0, t1, z, step=6.0):
    """Points of an arc of radius r from t0 to t1 (degrees), at height z."""
    t = np.radians(np.linspace(t0, t1, max(2, int(abs(t1 - t0) / step) + 1)))
    return np.column_stack([r * np.cos(t), r * np.sin(t), np.full(len(t), z)])


def _electronics(add) -> Frame:
    """The computer (behind), the driver board (front left), the cables between them and from the motors.
    Returns the frame of the driver board."""
    on_base = BASE_TOP + 0.0019
    pi = _board(add, (0.0, 0.0945), 90, 0.065, 0.030, "pcb")
    add("world", "chip", box(pi, -0.006, 0.006, -0.006, 0.006, 0, 0.0012))
    add("world", "chip", box(pi, -0.031, -0.0265, -0.0085, 0.0085, 0, 0.0014))            # camera connector
    add("world", "zinc", box(pi, 0.004, 0.0115, -0.0155, -0.0105, 0, 0.0026))             # USB
    add("world", "zinc", box(pi, 0.017, 0.0245, -0.0155, -0.0105, 0, 0.0026))
    add("world", "zinc", box(pi, -0.018, -0.007, -0.0155, -0.0085, 0, 0.0032))            # HDMI
    add("world", "black", box(pi, -0.0255, 0.0255, 0.0085, 0.0135, 0, 0.0025))            # 2 x 20 header
    for k in range(20):
        for row in (0.0098, 0.0122):
            pin = pi.at(-0.0241 + k * 0.00254, row, 0)
            add("world", "gold", box(pin, -0.0003, 0.0003, -0.0003, 0.0003, 0.0025, 0.0085))
    add("world", "led_status", box(pi, 0.028, 0.0295, 0.009, 0.0105, 0, 0.0006))
    a = math.radians(210)
    drivers = _board(add, (0.096 * math.cos(a), 0.096 * math.sin(a)), 300, 0.05, 0.04, "pcb")
    for k in range(3):
        x = -0.016 + 0.016 * k
        add("world", "black", box(drivers, x - 0.0065, x + 0.0065, -0.0095, 0.0095, 0, 0.0085))
        module = drivers.at(x, 0, 0.0085)
        add("world", "pcb_purple", box(module, -0.0076, 0.0076, -0.0102, 0.0102, 0, 0.0012))
        add("world", "alu", box(module, -0.0045, 0.0045, -0.0045, 0.0045, 0.0012, 0.0024))
        for j in range(4):
            fx = -0.0037 + j * 0.0024
            add("world", "alu", box(module, fx - 0.0004, fx + 0.0004, -0.0045, 0.0045, 0.0024, 0.0072))
        add("world", "white_plastic", box(drivers, x - 0.005, x + 0.005, 0.0135, 0.0195, 0, 0.0055))
    add("world", "capacitor", cylinder(drivers.at(-0.012, -0.015, 0), 0.004, 0.0, 0.0088, 24))
    add("world", "zinc", disc(drivers.at(-0.012, -0.015, 0), 0.004, 0.00882, True, 0, 24))
    add("world", "terminal", box(drivers, 0.005, 0.017, -0.0195, -0.0115, 0, 0.0085))
    for x in (0.008, 0.014):
        add("world", "zinc", cylinder(drivers.at(x, -0.0155, 0), 0.0016, 0.0085, 0.0090, 12))
    # motor cables: on a ring around the camera (out of reach of the arms), then to the connectors of the drivers
    ring = 0.0215
    for i, (t0, t1, r) in enumerate(((17, -150, ring - 0.0012), (137, 210, ring - 0.0012), (257, 210, ring + 0.0016))):
        a = math.radians(120 * i)
        u, m = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        start = (0.05 - 0.0295) * u + 0.012 * m
        x = (-0.016, 0.0, 0.016)[i]
        end = drivers.p(x, 0.028, 0)
        pts = np.concatenate([[[start[0], start[1], on_base]], _ring(r, t0, t1, on_base, 12),
                              [[end[0], end[1], on_base], drivers.p(x, 0.0215, 0.0), drivers.p(x, 0.0195, 0.0027)]])
        add("world", "cable", tube(smooth_path(pts, 6), 0.0018, 12, caps=False))
    # the computer to the drivers (signals and 5 V)
    gray = BASE_TOP + 0.0013
    pts = np.concatenate([[pi.p(0.0, 0.0115, 0.009), pi.p(-0.003, 0.0115, 0.013), pi.p(-0.02, 0.016, 0.0),
                           (-0.012, 0.040, gray), (-0.0115, 0.027, gray)], _ring(ring + 0.004, 115, 205, gray, 11),
                          [drivers.p(-0.026, 0.026, -0.0042 + 0.0013), drivers.p(-0.026, 0.012, 0.0045),
                           drivers.p(-0.021, 0.006, 0.0095)]])
    add("world", "cable_gray", tube(smooth_path(pts, 6), 0.0012, 10))
    return drivers


def _battery(add, drivers: Frame):
    """A 3-cell LiPo battery (73 x 35 x 23 mm) strapped on the base, front right; its leads go to the drivers."""
    a = math.radians(325)
    B = Frame((0.100 * math.cos(a), 0.100 * math.sin(a), BASE_TOP), (-math.sin(a), math.cos(a), 0),
              (-math.cos(a), -math.sin(a), 0), (0, 0, 1))      # e1 along the edge of the base, e2 inward
    section = [(x, y + 0.0115) for x, y in rounded_rectangle(0.0175, 0.0115, 0.0035)]
    add("world", "lipo", prism(Frame(B.o, B.e2, B.e3, B.e1), section, -0.0365, 0.0365, smooth=True))
    add("world", "lipo_label", box(B, -0.004, 0.004, -0.0150, 0.0150, 0.0230, 0.02306))
    for t in (-0.020, 0.020):                                  # hook-and-loop straps
        add("world", "strap", box(B, t - 0.0065, t + 0.0065, -0.0182, 0.0182, 0.0230, 0.0236))
        add("world", "strap", box(B, t - 0.0065, t + 0.0065, -0.0188, -0.0175, 0.0, 0.0236))
        add("world", "strap", box(B, t - 0.0065, t + 0.0065, 0.0175, 0.0188, 0.0, 0.0236))
    end = -0.0365
    for k, material in enumerate(("wire_red", "cable")):       # power leads, to the XT60 plug (yellow)
        off = 0.004 * (k - 0.5)
        lead = [B.p(end, off, 0.009), B.p(end - 0.008, off, 0.008), B.p(end - 0.014, off * 0.5, 0.005)]
        add("world", material, tube(smooth_path(lead, 6), 0.0011, 10))
    add("world", "xt60", box(B, end - 0.030, end - 0.014, -0.004, 0.004, 0.0002, 0.0082))
    add("world", "xt60", box(B, end - 0.046, end - 0.030, -0.004, 0.004, 0.0002, 0.0082))
    add("world", "white_plastic", box(B, end - 0.010, end - 0.004, 0.009, 0.0135, 0.0002, 0.0045))   # balance plug
    for k, material in enumerate(("cable", "wire_red", "cable_gray", "cable_gray")):
        off = 0.0101 + 0.0011 * k
        add("world", material, tube(smooth_path([B.p(end, off, 0.004 + 0.001 * k), B.p(end - 0.0035, off, 0.0028)], 4),
                                    0.00045, 6))
    start = B.p(end - 0.046, 0.0, 0.004)
    t0 = math.degrees(math.atan2(start[1], start[0])) % 360
    z = BASE_TOP + 0.0009
    for k, material in enumerate(("wire_red", "cable")):       # to the terminal of the driver board, along the edge
        off = 0.0014 * (k - 0.5)
        x = 0.008 + 0.006 * k
        pts = np.concatenate([[start + off * B.e2, B.p(end - 0.052, off, 0.0012)],
                              _ring(0.101 + off, t0 - 4, 258, z), _ring(0.113 + off, 249, 228, z),
                              [drivers.p(x, -0.026, -0.0042 + 0.0009), drivers.p(x, -0.0215, 0.002),
                               drivers.p(x, -0.0175, 0.0075)]])
        add("world", material, tube(smooth_path(pts, 6), 0.0008, 8))


def _panel(add):
    """Front panel (black anodized): the name, two LEDs PID and AI with their engraved labels."""
    F = Frame((0, -0.121, BASE_TOP), (1, 0, 0), (0, 0, 1), (0, -1, 0))   # e3 faces the default view
    add("world", "black", box(F, -0.024, 0.024, 0, 0.026, -0.0015, 0.0015))
    add("world", "black", box(F, -0.024, 0.024, 0, 0.0018, -0.012, -0.0015))
    for x in (-0.016, 0.016):
        socket_head(add, "world", Frame(F.p(x, 0, -0.0075), (1, 0, 0), (0, 1, 0), (0, 0, 1)), z0=0.0018, h=0.0018)
    add("world", "engraving", text_flat(Frame(F.p(0, 0.0215, 0.00156), F.e1, F.e2, F.e3), NAME, 0.0045))
    for x, led in ((-0.011, "pid"), (0.011, "ai")):
        L = Frame(F.p(x, 0.0135, 0.0015), F.e1, F.e2, F.e3)
        add("world", "chrome", cylinder(L, 0.0031, 0.0, 0.0008, 32))
        add("world", f"led_{led}", lathe(L, [(0, 0.0008), (0.0026, 0.0008), (0.0026, 0.0011), (0.0023, 0.0011),
                                             (0.0023, 0.0042)], 32))
        add("world", f"led_{led}", lathe(L, [(0.0023 * math.cos(t), 0.0042 + 0.0023 * math.sin(t))
                                             for t in np.linspace(0, math.pi / 2, 7)], 32, smooth=True))
        label = Frame(F.p(x, 0.0050, 0.00156), F.e1, F.e2, F.e3)
        if led == "pid":
            add("world", "engraving", text_flat(label, "PID", 0.0042))
        else:
            for code in LANGUAGES:
                add("world", "engraving", text_flat(label, load(code)["robot.ai"], 0.0042), f"language:{code}")


def _ball(add, radius):
    """Seam of the ping-pong ball; one printing per ball, on the side seen in the default view at rest."""
    w = Frame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    add("ball", "seam", lathe(w, [(radius - 0.0001, -0.00045), (radius + 0.00009, -0.00018),
                                  (radius + 0.00009, 0.00018), (radius - 0.0001, 0.00045)], 96, smooth=True))
    for name, lines in PRINTS.items():
        directions = ((0, -1, 0.55),) if len(lines) == 1 else ((0, -1, 0.75), (0, -1, 0.33))
        sizes = (0.0040,) if len(lines) == 1 else (0.0036, 0.0024)
        for text, d, h in zip(lines, directions, sizes):
            add("ball", "print", text_on_sphere(radius, d, (0, 0, 1), text, h), f"ball:{name}")
