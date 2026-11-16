"""The robot as shown in the window: a copy of the MuJoCo model of the physics, dressed for display.

The display model has the same bodies and joints as the physics model (same qpos), with materials, lights, a sky,
a floor, and the parts of a lab prototype (app/parts.py) instead of the simple shapes of the physics, which stay in it
hidden (group HIDDEN): same bodies, same joints, same shapes, so the picture is carried by the simulated mechanism.
At every frame, the positions of the physics are copied into it (qpos) and its geometry is computed (no dynamics):
it never acts on the physics, and the virtual camera, which renders the physics model, keeps seeing the scene it has
always seen. Some parts depend on what the robot does (DisplayScene.show): the LED of its brain is lit, the ball
carries the printing of the ball in play, the label of the AI is in the language of the interface. Without Qt.

Shadows and fog follow the view (DisplayScene.view): the shadow map covers the floor around the point looked at, and
it and the fog grow with the distance of the camera. In the default view (VIEW_DISTANCE), they are those of the
model: sharp shadows, a floor without dots, a horizon that fades into the sky. Farther (zoomed out, or following a
fallen ball), all their lengths grow in proportion, so the floor seen keeps the same look.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

import mujoco

from app.parts import DEFAULTS, LED_ON, MATERIALS, robot_meshes
from simulation.config import Config
from simulation.mujoco_physics import model_xml

MAX_SIZE = (2560, 1600)         # (width, height) of the largest image rendered, in pixels
# Quality, from the measurements of app.benchmark (Intel Arc integrated graphics): antialiasing with 8 samples costs
# 13 to 46 % more than with 4; a shadow map of 2048 instead of 4096, or no fog, saves nothing measurable.
ANTIALIASING = 4                # samples
SHADOW_MAP = 4096               # (pixels) side of the map of the shadows
# Shadows and fog in the default view, in sizes of the scene (MuJoCo: × stat.extent, 1.2 m here). Beyond the floor
# covered by the shadow map, the floor gets speckled; a larger map has larger pixels, which shows as steps on the
# edges of the shadows (measured: 6 m wide, 2.9 mm per pixel of the map). A nearer fog hides the far floor sooner.
VIEW_DISTANCE = 0.68            # (m) distance of the camera in the default view (app/view3d.Camera)
SHADOW_CLIP = 0.6               # half-width of the floor covered by the shadow map: 0.72 m, 0.35 mm per pixel of it
FOG = (1.0, 3.0)                # start and end of the fog, from the eye: 1.2 to 3.6 m
HIDDEN = 3                      # group of geoms that MuJoCo does not draw (by default, it draws groups 0 to 2)
LED_OFF = 0.32                  # an LED off: its color dimmed


@dataclass(frozen=True)
class SceneStyle:
    """Colors (r, g, b) in 0..1 of the 3D scene, for one theme."""
    sky: tuple                    # also the color of the fog: the floor fades into the sky
    floor: tuple
    plate: tuple                  # acrylic
    plate_alpha: float
    ball: tuple = (1.0, 0.5, 0.0)  # ping-pong orange
    ambient: float = 0.4          # light coming from the point of view (headlight)


def _rgb(c) -> str:
    return " ".join(f"{v:.3f}" for v in c)


def _parts(cfg: Config):
    """The parts of the robot of cfg: {body: [geom]}, their meshes {file: bytes}, and the geoms that depend on a
    choice {geom name: (kind, value)}. Built once per robot geometry."""
    return _parts_of(cfg.geometry, cfg.plate.plate_radius, cfg.mujoco_params.plate_thickness, cfg.plate.ball_radius)


@lru_cache(maxsize=4)
def _parts_of(geometry, plate_radius: float, plate_thickness: float, ball_radius: float):
    meshes = robot_meshes(geometry, plate_radius, plate_thickness, ball_radius)
    geoms, files, switches = {}, {}, {}
    for (body, material, switch), mesh in meshes.items():
        kind, _, value = switch.partition(":")
        name = f"{body}_{material}" + (f"_{value}" if switch else "")
        files[f"{name}.msh"] = mesh.msh()
        if switch:
            switches[f"deco_{name}"] = (kind, value)
        shown = not switch or DEFAULTS[kind] == value
        geoms.setdefault(body, []).append(
            f'<geom name="deco_{name}" type="mesh" mesh="{name}" material="part_{material}" contype="0" '
            f'conaffinity="0" group="{0 if shown else HIDDEN}"/>')
    return geoms, files, switches


def _led(on: bool):
    """Color (r, g, b, a) and emission of an LED: lit, or off (its color dimmed, the lens seen through)."""
    if on:
        return (*LED_ON, 1.0), 1.0
    return (*(LED_OFF * c for c in LED_ON), 0.92), 0.0


def display_assets(cfg: Config) -> dict:
    """The meshes of the parts, {file: bytes}, for mujoco.MjModel.from_xml_string(display_xml(...), assets)."""
    return _parts(cfg)[1]


def display_xml(cfg: Config, style: SceneStyle) -> str:
    """The MJCF of the physics model of cfg, dressed with `style` (its meshes: display_assets). Only the appearance
    and the fixed parts change: same bodies, joints and shapes of the physics (those of the moving parts, hidden)."""
    s = style
    floor_size = 6.0
    geoms, files, _ = _parts(cfg)
    materials = [f'<material name="part_{name}" rgba="{_rgb(m[0])} 1" specular="{m[1]}" shininess="{m[2]}"'
                 + (f' emission="{m[3]}"' if len(m) > 3 else "") + "/>" for name, m in MATERIALS.items()]
    materials.append(f'<material name="part_acrylic_edge" rgba="{_rgb(s.plate)} 0.75" specular="1" shininess="1"/>')
    for led in ("pid", "ai"):                       # the PID one lit until DisplayScene.show says otherwise
        rgba, emission = _led(led == "pid")
        materials.append(f'<material name="part_led_{led}" rgba="{_rgb(rgba)}" specular="1" shininess="1" '
                         f'emission="{emission}"/>')
    meshes = [f'<mesh name="{file[:-4]}" file="{file}" inertia="shell"/>' for file in files]
    visual = f"""<visual>
    <global azimuth="90" elevation="-30" offwidth="{MAX_SIZE[0]}" offheight="{MAX_SIZE[1]}"/>
    <quality shadowsize="{SHADOW_MAP}" offsamples="{ANTIALIASING}"/>
    <headlight ambient="{_rgb([s.ambient] * 3)}" diffuse="{_rgb([0.8 * s.ambient] * 3)}" specular="0.05 0.05 0.05"/>
    <map fogstart="{FOG[0]}" fogend="{FOG[1]}" znear="0.01" zfar="20" haze="0.4" shadowclip="{SHADOW_CLIP}"/>
    <rgba fog="{_rgb(s.sky)} 1" haze="{_rgb(s.sky)} 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{_rgb(s.sky)}" rgb2="{_rgb(s.sky)}" width="32" height="192"/>
    <material name="floor" rgba="{_rgb(s.floor)} 1" specular="0.1" shininess="0.2"/>
    <material name="plate" rgba="{_rgb(s.plate)} {s.plate_alpha}" specular="0.9" shininess="0.95"/>
    <material name="ball" rgba="{_rgb(s.ball)} 1" specular="0.5" shininess="0.6"/>
    """ + "\n    ".join(materials + meshes) + """
  </asset>"""
    lights = ('<light directional="true" pos="0.3 -0.4 1.5" dir="-0.2 0.3 -1" diffuse="0.45 0.45 0.45" '
              'specular="0.3 0.3 0.3" castshadow="true"/>\n'
              '    <light directional="true" pos="-0.8 0.6 1" dir="0.5 -0.4 -0.6" diffuse="0.15 0.15 0.15" '
              'specular="0 0 0" castshadow="false"/>')

    def after(body: str, indent: str):
        """Replacement: the geom of the physics, then the parts of `body`."""
        return lambda m: m.group(0) + "".join(f"\n{indent}{g}" for g in geoms[body])
    replacements = [
        (r'<geom name="mount\d" type="box" size="[^"]*" pos="[^"]*" rgba="[^"]*"/>\s*', ""),
        (r'<visual><global azimuth="120" elevation="-30"/></visual>', visual),
        (r'<light pos="0 0 1.5" dir="0 0 -1"/>', lights),
        (r'<geom name="ground" type="plane" size="0.4 0.4 0.01" pos="0 0 -0.04" rgba="[^"]*"/>',
         f'<geom name="ground" type="plane" size="{floor_size} {floor_size} 0.05" pos="0 0 -0.04" '
         f'material="floor"/>' + "".join(f"\n    {g}" for g in geoms["world"])),
        (r'rgba="0.7 0.85 1 0.5"', 'material="plate"'),
        (r'rgba="1 0.5 0 1"', 'material="ball"'),
        (r'rgba="0.2 0.4 0.8 1"', f'rgba="0.2 0.4 0.8 1" group="{HIDDEN}"'),       # arms of the physics
        (r'rgba="0.55 0.55 0.6 1"', f'rgba="0.55 0.55 0.6 1" group="{HIDDEN}"'),   # rods of the physics
    ] + [(rf'<body name="arm{i}" pos="[^"]*">\s*<joint [^>]*/>\s*<geom [^>]*/>', after(f"arm{i}", "      "))
         for i in (1, 2, 3)] + \
        [(rf'<body name="rod{i}" pos="[^"]*">\s*<joint [^>]*/>\s*<geom [^>]*/>', after(f"rod{i}", "        "))
         for i in (1, 2, 3)] + \
        [(r'<geom name="plate" [^>]*/>', after("plate", "      ")),
         (r'<geom name="ball" type="sphere"[^>]*/>', after("ball", "      "))]
    xml = model_xml(cfg)
    for pattern, new in replacements:
        xml, n = re.subn(pattern, new if callable(new) else (lambda _, new=new: new), xml)
        if n == 0:
            raise ValueError(f"display_xml: the physics model has changed, '{pattern}' not found")
    return xml


class DisplayScene:
    """The display model and its data, kept in step with the physics data."""

    def __init__(self, cfg: Config, style: SceneStyle):
        self.model = mujoco.MjModel.from_xml_string(display_xml(cfg, style), display_assets(cfg))
        self.data = mujoco.MjData(self.model)
        self._light_pos = self.model.light_pos.copy()
        self._switches = {}                 # {(kind, value): [geom]}: the parts that depend on a choice
        for name, key in _parts(cfg)[2].items():
            self._switches.setdefault(key, []).append(self.model.geom(name).id)
        self._leds = {led: self.model.material(f"part_led_{led}").id for led in ("pid", "ai")}
        self._shown = None
        self._ground = self.model.geom("ground").id

    def prepare(self, scene: mujoco.MjvScene) -> None:
        """After mujoco.mjv_updateScene, before rendering: fog on, and the floor casts no shadow (it still receives
        those of the robot). MuJoCo draws into the shadow map every geom but the decorations; the floor drawn there
        shadowed itself in dots wherever the camera saw it beyond the map, a square 1.44 m wide around the point looked
        at in the default view (measured at azimuth 0°: 0.43 % of the floor seen in dots, in a region bounded by a
        corner of the square; none without)."""
        scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        for j in range(scene.ngeom):
            g = scene.geoms[j]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == self._ground:
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                break

    def show(self, brain: str, ball: str, language: str) -> None:
        """The parts that depend on what the robot does: the LED of its brain lit (PID, or AI for any other), the
        printing of the ball in play (that of the ping-pong ball for a ball of no name), the label of the AI in the
        language of the interface. Once per frame: changes the model only when one of them changes."""
        if self._shown == (brain, ball, language):
            return
        self._shown = (brain, ball, language)
        chosen = {"ball": ball if ("ball", ball) in self._switches else DEFAULTS["ball"],
                  "language": language if ("language", language) in self._switches else DEFAULTS["language"]}
        for (kind, value), geoms in self._switches.items():
            self.model.geom_group[geoms] = 0 if value == chosen[kind] else HIDDEN
        lit = "pid" if brain == "pid" else "ai"
        for led, material in self._leds.items():
            self.model.mat_rgba[material], self.model.mat_emission[material] = _led(led == lit)

    def view(self, context, lookat, distance: float) -> None:
        """Adapts the shadows and the fog to the camera, before rendering: those of the default view, all lengths
        multiplied by scale = distance / VIEW_DISTANCE (at least 1), the shadow map centered on the point looked at.
        context: the MuJoCo rendering context (mujoco.MjrContext), which holds them in meters.
        The lights are directional: moving them moves only the floor covered by the shadow map, not the light."""
        scale = max(1.0, distance / VIEW_DISTANCE)
        extent = self.model.stat.extent
        context.shadowClip = SHADOW_CLIP * extent * scale
        context.fogStart, context.fogEnd = FOG[0] * extent * scale, FOG[1] * extent * scale
        self.model.light_pos[:] = self._light_pos + (lookat[0], lookat[1], 0.0)

    def follow(self, physics_data) -> None:
        """Copies the positions of the physics and computes the geometry, lights included (no dynamics)."""
        self.data.qpos[:] = physics_data.qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
