"""What the software shows, without Qt: the display scene (app/scene.py) and the parts of the robot (app/parts.py),
the themes (app/theme.py) and the measurements of the panel (app/readouts.py)."""
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from app.meshes import font_file
from app.parts import BASE_TOP, EYE_DEPTH, GROUND, PRINTS, robot_meshes
from app.readouts import distance_history, readouts
from app.scene import (ANTIALIASING, FOG, HIDDEN, SHADOW_CLIP, SHADOW_MAP, VIEW_DISTANCE, DisplayScene, display_assets,
                       display_xml)
from app.theme import COLORS, MODES, scene_style, stylesheet
from simulation.mujoco_physics import model_xml
from simulation.session import BALLS, Session, sandbox_config


def advance(s: Session, duration: float) -> None:
    for _ in range(round(duration / s.dt)):
        s.step()


def test_display_scene_shows_the_simulated_robot():
    s = Session(sandbox_config(), record=False)
    s.throw(0.10, 180, speed=0.4)
    advance(s, 0.5)
    scene = DisplayScene(s.cfg, scene_style("light"))
    scene.follow(s.phys.data)
    physics = mujoco.MjData(s.phys.model)                 # the geometry of the physics at the same instant
    physics.qpos[:] = s.phys.data.qpos
    mujoco.mj_kinematics(s.phys.model, physics)
    m, d = s.phys.model, scene.model
    assert d.nq == m.nq and d.nbody == m.nbody
    assert (d.vis.quality.offsamples, d.vis.quality.shadowsize) == (ANTIALIASING, SHADOW_MAP)
    physics_geoms = lambda model, b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                                      and not model.geom(g).name.startswith("deco_")]
    for b in range(1, m.nbody):                           # every moving part: same place, same shapes
        assert d.body(b).name == m.body(b).name
        assert np.allclose(scene.data.xpos[b], physics.xpos[b], rtol=0, atol=1e-12)
        assert np.allclose(scene.data.xmat[b], physics.xmat[b], rtol=0, atol=1e-12)
        shapes = lambda model: sorted((int(model.geom_type[g]), tuple(np.round(model.geom_size[g], 9)))
                                      for g in physics_geoms(model, b))
        assert shapes(d) == shapes(m)
        if d.body(b).name.startswith(("arm", "rod")):     # their simple shapes hidden, the parts shown instead
            assert all(d.geom_group[g] == HIDDEN for g in physics_geoms(d, b))
    parts = [g for g in range(d.ngeom) if d.geom(g).name.startswith("deco_")]
    assert len(parts) > 50 and d.ngeom == m.ngeom - 3 + len(parts)            # the 3 blocks of the motors replaced
    assert all(d.geom_contype[g] == 0 and d.geom_conaffinity[g] == 0 for g in parts)   # the parts touch nothing


def test_parts_on_the_joints():
    """The arm is a link from the motor axis to the elbow; the lower rod end turns about the elbow; the upper one sits
    EYE_DEPTH below the ball joint of the physics, on the bolt of the bracket under the plate (at rest: the give of
    the simulated ball joint apart, 0.1 mm)."""
    s = Session(sandbox_config(), record=False)
    g = s.cfg.geometry
    scene = DisplayScene(s.cfg, scene_style("light"))
    scene.follow(s.phys.data)
    m, d = scene.model, scene.data

    def vertices(name, body):
        """Vertices of a part in the frame of its body."""
        k = m.geom(name).id
        mid = m.geom_dataid[k]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        world = d.geom_xpos[k] + v @ d.geom_xmat[k].reshape(3, 3).T
        return (world - d.body(body).xpos) @ d.body(body).xmat.reshape(3, 3)
    plate = vertices("deco_plate_alu", "plate")[:, 2]                     # the three brackets
    assert abs(plate.min() + EYE_DEPTH + 0.008) < 1e-4 and abs(plate.max() + 0.003) < 1e-4
    for i in range(3):
        u = np.array([*g.direction(i), 0.0])
        link = vertices(f"deco_arm{i + 1}_alu", f"arm{i + 1}") @ u
        assert link.min() < -0.009 and abs(link.max() - (g.arm_length + 0.0068)) < 1e-4
        rod = vertices(f"deco_rod{i + 1}_zinc", f"rod{i + 1}")[:, 2]       # rod ends: eyes of radius 6.5 mm
        assert abs(rod.min() + 0.0065) < 1e-4 and abs(rod.max() - (g.rod_length - EYE_DEPTH + 0.0065)) < 1e-4
        r, p = d.body(f"rod{i + 1}"), d.body("plate")
        eye = r.xpos + r.xmat.reshape(3, 3) @ [0, 0, g.rod_length - EYE_DEPTH]
        bolt = p.xpos + p.xmat.reshape(3, 3) @ ((g.base_radius + g.arm_length) * u - [0, 0, EYE_DEPTH])
        assert np.linalg.norm(eye - bolt) < 2e-4


def test_moving_parts_clear_the_base():
    """Hard catch: an arm goes down by 28°, its lowest part stays above the base (measured: 1.7 mm)."""
    s = Session(sandbox_config(), record=False)
    scene = DisplayScene(s.cfg, scene_style("light"))
    m = scene.model
    moving = [k for k in range(m.ngeom) if m.geom(k).name.startswith(("deco_arm", "deco_rod"))]
    s.throw(0.12, 200, speed=0.9)
    low = 1.0
    for k in range(1200):
        s.step()
        if k % 10 == 0:
            scene.follow(s.phys.data)
            for g in moving:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                low = min(low, float((scene.data.geom_xpos[g][2] + v @ scene.data.geom_xmat[g].reshape(3, 3)[2]).min()))
    assert BASE_TOP + 0.001 < low < BASE_TOP + 0.003


def test_parts_shown_by_what_the_robot_does():
    """The LED of the brain lit, the printing of the ball in play, the label of the AI in the language."""
    s = Session(sandbox_config(), record=False)
    scene = DisplayScene(s.cfg, scene_style("dark"))
    m = scene.model
    shown = lambda name: m.geom_group[m.geom(name).id] != HIDDEN
    led = lambda name: m.mat_emission[m.material(f"part_led_{name}").id]
    assert shown("deco_ball_print_ping_pong") and shown("deco_world_engraving_en")     # before any choice
    assert not shown("deco_ball_print_solid") and not shown("deco_world_engraving_fr")
    scene.show("imitation", "solid", "fr")
    assert shown("deco_ball_print_solid") and not shown("deco_ball_print_ping_pong")
    assert shown("deco_world_engraving_fr") and not shown("deco_world_engraving_en")
    assert (led("ai"), led("pid")) == (1.0, 0.0)
    scene.show("pid", "other", "de")                       # a ball and a language of no name: the defaults
    assert shown("deco_ball_print_ping_pong") and shown("deco_world_engraving_en") and (led("pid"), led("ai")) == (1, 0)
    assert set(PRINTS) == set(BALLS) and shown("deco_world_engraving")                # name, PID: always


def test_parts_built_once():
    """The meshes of the parts are built once (0.35 s on the development computer), then shared by every scene
    (themes, side by side)."""
    s = Session(sandbox_config(), record=False)
    DisplayScene(s.cfg, scene_style("light"))
    before = robot_meshes.cache_info()
    DisplayScene(s.cfg, scene_style("dark"))
    after = robot_meshes.cache_info()
    assert after.misses == before.misses
    assert all(file.endswith(".msh") for file in display_assets(s.cfg))
    assert s.phys.model.geom("ground").pos[2] == GROUND                               # the parts stand on its floor


def test_font_of_the_texts():
    """The texts of the parts (name, labels, printings) are drawn with the font file that the font manager of
    Matplotlib finds for DejaVu Sans Bold: that shipped with Matplotlib, read directly (app/meshes.font_file)."""
    from matplotlib import font_manager
    found = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
                                  fallback_to_default=False)
    assert Path(found).resolve() == font_file().resolve()


def test_display_scene_leaves_the_physics_alone():
    s = Session(sandbox_config(), record=False)
    xml = model_xml(s.cfg)
    for mode in MODES:
        assert display_xml(s.cfg, scene_style(mode)) != xml
        DisplayScene(s.cfg, scene_style(mode)).follow(s.phys.data)
    assert model_xml(s.cfg) == xml
    assert s.phys.model.vis.global_.offwidth == mujoco.MjModel.from_xml_string(xml).vis.global_.offwidth


def dots(image) -> float:
    """Share of the pixels (%) that differ by more than 12 levels from the mean of their 4 neighbors."""
    g = image.astype(float).mean(axis=2)
    neighbors = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4
    return 100 * float((np.abs(g[1:-1, 1:-1] - neighbors) > 12).mean())


def test_shadows_and_fog_follow_the_view():
    """Default view: the shadows and the fog of the model, the image does not change. Farther: their lengths grow with
    the distance of the camera, the shadow map is centered on the point looked at, and the floor keeps its look
    (measured at 480 × 300: dots on the floor 4.1 % without this, 0.4 % with it, following a fallen ball)."""
    s = Session(sandbox_config(), record=False)
    s.phys.place_ball(1.25, 0.3, 0.0)                     # far from the robot, as a fallen ball
    scene = DisplayScene(s.cfg, scene_style("light"))
    renderer = mujoco.Renderer(scene.model, 300, 480)
    context, extent, lights = renderer._mjr_context, scene.model.stat.extent, scene.model.light_pos.copy()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90.0, -27.0, VIEW_DISTANCE
    camera.lookat[:] = (0.0, 0.0, s.cfg.geometry.height - 0.035)

    def render():
        scene.follow(s.phys.data)
        renderer.update_scene(scene.data, camera=camera)
        scene.prepare(renderer.scene)
        return renderer.render().copy()
    before = render()
    scene.view(context, camera.lookat, camera.distance)
    assert np.array_equal(render(), before)
    assert context.shadowClip == np.float32(SHADOW_CLIP * extent)
    assert (context.fogStart, context.fogEnd) == (np.float32(FOG[0] * extent), np.float32(FOG[1] * extent))
    camera.lookat[:2], camera.distance = (0.62, 0.15), 3 * VIEW_DISTANCE     # following the fallen ball
    scene.view(context, camera.lookat, camera.distance)
    assert context.shadowClip == np.float32(3 * SHADOW_CLIP * extent)
    assert context.fogEnd == np.float32(3 * FOG[1] * extent)
    image = render()
    assert np.allclose(scene.data.light_xpos, lights + (0.62, 0.15, 0.0))
    assert dots(image) < 1.0
    renderer.close()


def test_floor_casts_no_shadow():
    """Turned to azimuth 0°: no dots on the floor beyond the shadow map (measured: 0.43 % of the floor seen in dots
    when it cast its own shadow, in a region bounded by a corner of the map; none now). The robot's shadows stay."""
    s = Session(sandbox_config(), record=False)
    scene = DisplayScene(s.cfg, scene_style("light"))
    renderer = mujoco.Renderer(scene.model, 300, 480)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 0.0, -27.0, VIEW_DISTANCE
    camera.lookat[:] = (0.0, 0.0, s.cfg.geometry.height - 0.035)
    scene.view(renderer._mjr_context, camera.lookat, camera.distance)
    scene.follow(s.phys.data)

    def render(prepared: bool):
        renderer.update_scene(scene.data, camera=camera)
        if prepared:
            scene.prepare(renderer.scene)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        return renderer.render().astype(float).mean(axis=2)
    image, casting = render(True), render(False)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(scene.data, camera=camera)
    ids = renderer.render()
    renderer.close()
    floor = (ids[..., 0] == scene.model.geom("ground").id) & (ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    inner = floor[1:-1, 1:-1] & floor[:-2, 1:-1] & floor[2:, 1:-1] & floor[1:-1, :-2] & floor[1:-1, 2:]

    def floor_dots(g):
        """Share (%) of the floor in dots: pixels darker by 12 levels than each of their 4 neighbors (the edges
        of the shadows, lines, are not)."""
        darkest = np.minimum.reduce([g[:-2, 1:-1], g[2:, 1:-1], g[1:-1, :-2], g[1:-1, 2:]])
        return 100 * float(((g[1:-1, 1:-1] < darkest - 12) & inner).sum() / inner.sum())
    assert floor_dots(casting) > 0.2 and floor_dots(image) == 0.0
    assert (image[floor] < image[floor].max() - 40).sum() > 1000                   # the shadow of the robot


def lightness(image):
    """CIE L* (0 black to 100 white) of sRGB pixels."""
    c = image / 255.0
    y = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, y * 24389 / 27)


def test_the_plate_stands_out_in_both_themes():
    """Seen through the plate, the scene changes in lightness by 18 L* at least, in both themes (measured: 23 in
    the light theme with the smoked plate, 13 with the clear one it replaced; 29 in the dark theme)."""
    s = Session(sandbox_config(), record=False)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90.0, -27.0, VIEW_DISTANCE
    camera.lookat[:] = (0.0, 0.0, s.cfg.geometry.height - 0.035)

    def render(style):
        scene = DisplayScene(s.cfg, style)
        scene.follow(s.phys.data)
        renderer = mujoco.Renderer(scene.model, 250, 350)
        renderer.update_scene(scene.data, camera=camera)
        image = renderer.render().astype(float)
        renderer.close()
        return image
    for mode in MODES:
        style = scene_style(mode)
        without = render(replace(style, plate_alpha=0.0))
        plate = np.any(np.abs(render(replace(style, plate=(1.0, 0.0, 1.0), plate_alpha=1.0)) - without) > 1, axis=2)
        change = np.abs(lightness(render(style)) - lightness(without))[plate]
        assert plate.sum() > 5000 and change.mean() > 18, mode


def test_themes():
    assert COLORS["light"].keys() == COLORS["dark"].keys()
    for mode in MODES:
        sheet = stylesheet(mode)
        assert COLORS[mode]["accent"] in sheet and COLORS[mode]["bg"] in sheet
        assert len(scene_style(mode).sky) == 3


def test_readouts():
    s = Session(sandbox_config(), record=False)
    s.center_ball()
    advance(s, 3.0)
    v = readouts(s)
    assert v["distance_mm"] < 5 and v["speed_mm_s"] < 20 and v["height_mm"] < 1
    assert abs(v["tilt_x_deg"]) < 10 and abs(v["tilt_y_deg"]) < 10 and v["torque_Nm"] > 0
    assert len(distance_history(s)) == 150                 # 3 s at 50 Hz
    s.throw(0.10, 180, speed=1.0)                          # too fast: falls
    advance(s, 3.0)
    v = readouts(s)
    assert s.state == "fallen"
    assert v["distance_mm"] is None and v["x_mm"] is None and v["tilt_x_deg"] is not None
