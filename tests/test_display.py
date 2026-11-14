"""What the software shows, without Qt: the display scene (app/scene.py), the themes (app/theme.py) and the
measurements of the panel (app/readouts.py)."""
import mujoco
import numpy as np

from app.readouts import distance_history, readouts
from app.scene import ANTIALIASING, SHADOW_MAP, DisplayScene, display_xml
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
