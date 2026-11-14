"""The robot as shown in the window: a copy of the MuJoCo model of the physics, dressed for display.

The display model has the same bodies and joints as the physics model (same qpos), with materials, lights, a sky,
a floor, a pedestal and stepper motors added. At every frame, the positions of the physics are copied into it (qpos)
and its geometry is computed (no dynamics): it never acts on the physics, and the virtual camera, which renders the
physics model, keeps seeing the scene it has always seen. The moving parts keep the sizes of the physics model: the
picture shows the robot that is simulated. Without Qt.
"""
from dataclasses import dataclass
import math
import re

import mujoco

from simulation.config import Config
from simulation.mujoco_physics import model_xml

MAX_SIZE = (2560, 1600)         # (width, height) of the largest image rendered, in pixels
# Quality, from the measurements of app.benchmark (Intel Arc integrated graphics): antialiasing with 8 samples costs
# 13 to 46 % more than with 4; a shadow map of 2048 instead of 4096, or no fog, saves nothing measurable.
ANTIALIASING = 4                # samples
SHADOW_MAP = 4096               # (pixels) side of the map of the shadows
SHADOW_CLIP = 5.0               # (m) half-width of the floor covered by the map; beyond, the floor gets speckled
                                # (measured: dots beyond 0.6 m with 0.6, none within 5 m with 5, where the camera
                                # can follow a fallen ball; 2.4 mm per pixel of the map, the shadows stay sharp)


@dataclass(frozen=True)
class SceneStyle:
    """Colors (r, g, b) in 0..1 of the 3D scene, for one theme."""
    sky: tuple                    # also the color of the fog: the floor fades into the sky
    floor: tuple
    pedestal: tuple
    motor: tuple
    metal: tuple                  # caps of the motors, hubs, rods
    arm: tuple
    plate: tuple                  # acrylic
    plate_alpha: float
    ball: tuple = (1.0, 0.5, 0.0)  # ping-pong orange
    ambient: float = 0.4          # light coming from the point of view (headlight)


def _rgb(c) -> str:
    return " ".join(f"{v:.3f}" for v in c)


def _motor(match) -> str:
    """A stepper motor (NEMA 17: 42 mm square, 40 mm long) on the axis of the arm i, behind it, on a bracket; with
    the hub of its shaft. Replaces the block that stands for it in the physics model (decoration, no contact)."""
    i, bx, by = match.group(1), float(match.group(2)), float(match.group(3))
    n = math.hypot(bx, by)
    ux, uy = bx / n, by / n
    ax, ay = uy, -ux                                         # axis of the arm (simulation/mujoco_physics.py)

    def at(d):
        return f"{bx - d * ax:.5f} {by - d * ay:.5f}"
    axis = f'xyaxes="{ux:.5f} {uy:.5f} 0 0 0 1"'              # local z = u × vertical = axis of the arm
    return (f'<geom name="motor{i}" type="box" size="0.021 0.021 0.014" pos="{at(0.03)} 0" {axis} '
            f'material="motor"/>\n'
            f'    <geom type="box" size="0.021 0.021 0.003" pos="{at(0.013)} 0" {axis} material="metal"/>\n'
            f'    <geom type="box" size="0.021 0.021 0.003" pos="{at(0.047)} 0" {axis} material="metal"/>\n'
            f'    <geom type="cylinder" size="0.0075 0.005" pos="{at(0.005)} 0" {axis} material="metal"/>\n'
            f'    <geom type="box" size="0.012 0.012 0.0035" pos="{at(0.03)} -0.0245" material="motor"/>')


def display_xml(cfg: Config, style: SceneStyle) -> str:
    """The MJCF of the physics model of cfg, dressed with `style`. Only the appearance and the fixed decoration
    change: same bodies, joints and sizes of the moving parts."""
    s = style
    floor_size = 6.0
    visual = f"""<visual>
    <global azimuth="90" elevation="-30" offwidth="{MAX_SIZE[0]}" offheight="{MAX_SIZE[1]}"/>
    <quality shadowsize="{SHADOW_MAP}" offsamples="{ANTIALIASING}"/>
    <headlight ambient="{_rgb([s.ambient] * 3)}" diffuse="{_rgb([0.8 * s.ambient] * 3)}" specular="0.05 0.05 0.05"/>
    <map fogstart="4" fogend="10" znear="0.01" zfar="30" haze="0.4" shadowclip="{SHADOW_CLIP}"/>
    <rgba fog="{_rgb(s.sky)} 1" haze="{_rgb(s.sky)} 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{_rgb(s.sky)}" rgb2="{_rgb(s.sky)}" width="32" height="192"/>
    <material name="floor" rgba="{_rgb(s.floor)} 1" specular="0.1" shininess="0.2"/>
    <material name="pedestal" rgba="{_rgb(s.pedestal)} 1" specular="0.3" shininess="0.5"/>
    <material name="motor" rgba="{_rgb(s.motor)} 1" specular="0.4" shininess="0.6"/>
    <material name="metal" rgba="{_rgb(s.metal)} 1" specular="0.9" shininess="0.8"/>
    <material name="arm" rgba="{_rgb(s.arm)} 1" specular="0.6" shininess="0.7"/>
    <material name="plate" rgba="{_rgb(s.plate)} {s.plate_alpha}" specular="0.9" shininess="0.95"/>
    <material name="ball" rgba="{_rgb(s.ball)} 1" specular="0.5" shininess="0.6"/>
  </asset>"""
    lights = ('<light directional="true" pos="0.3 -0.4 1.5" dir="-0.2 0.3 -1" diffuse="0.45 0.45 0.45" '
              'specular="0.3 0.3 0.3" castshadow="true"/>\n'
              '    <light directional="true" pos="-0.8 0.6 1" dir="0.5 -0.4 -0.6" diffuse="0.15 0.15 0.15" '
              'specular="0 0 0" castshadow="false"/>')
    pedestal = ('<geom name="pedestal" type="cylinder" size="0.14 0.006" pos="0 0 -0.034" material="pedestal"/>\n'
                '    <geom name="pedestal_foot" type="cylinder" size="0.12 0.002" pos="0 0 -0.0385" '
                'material="motor"/>')
    replacements = [
        (r'<geom name="mount(\d)" type="box" size="[^"]*" pos="([^ ]+) ([^ ]+) [^"]*" rgba="[^"]*"/>', _motor),
        (r'<visual><global azimuth="120" elevation="-30"/></visual>', visual),
        (r'<light pos="0 0 1.5" dir="0 0 -1"/>', lights),
        (r'<geom name="ground" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.04" rgba="[^"]*"/>',
         f'<geom name="ground" type="plane" size="{floor_size} {floor_size} 0.05" pos="0 0 -0.04" '
         f'material="floor"/>\n    {pedestal}'),
        (r'rgba="0.7 0.85 1 0.5"', 'material="plate"'),
        (r'rgba="1 0.5 0 1"', 'material="ball"'),
        (r'rgba="0.2 0.4 0.8 1"', 'material="arm"'),
        (r'rgba="0.55 0.55 0.6 1"', 'material="metal"'),
    ]
    xml = model_xml(cfg)
    for pattern, new in replacements:
        xml, n = re.subn(pattern, new if callable(new) else (lambda _, new=new: new), xml)
        if n == 0:
            raise ValueError(f"display_xml: the physics model has changed, '{pattern}' not found")
    return xml


class DisplayScene:
    """The display model and its data, kept in step with the physics data."""

    def __init__(self, cfg: Config, style: SceneStyle):
        self.model = mujoco.MjModel.from_xml_string(display_xml(cfg, style))
        self.data = mujoco.MjData(self.model)

    def follow(self, physics_data) -> None:
        """Copies the positions of the physics and computes the geometry, lights included (no dynamics)."""
        self.data.qpos[:] = physics_data.qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
