"""What the software shows, without Qt: the display scene (app/scene.py), the themes (app/theme.py) and the
measurements of the panel (app/readouts.py)."""
from dataclasses import replace

import mujoco
import numpy as np

from app.readouts import distance_history, readouts
from app.scene import ANTIALIASING, FOG, SHADOW_CLIP, SHADOW_MAP, VIEW_DISTANCE, DisplayScene, display_xml
from app.theme import COLORS, MODES, scene_style, stylesheet
from simulation.mujoco_physics import model_xml
from simulation.session import Session, sandbox_config


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
    for b in range(1, m.nbody):                           # every moving part: same place, same shapes
        assert d.body(b).name == m.body(b).name
        assert np.allclose(scene.data.xpos[b], physics.xpos[b], rtol=0, atol=1e-12)
        assert np.allclose(scene.data.xmat[b], physics.xmat[b], rtol=0, atol=1e-12)
        shapes = lambda model: sorted((int(model.geom_type[g]), tuple(np.round(model.geom_size[g], 9)))
                                      for g in range(model.ngeom) if model.geom_bodyid[g] == b)
        assert shapes(d) == shapes(m)


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
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
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
