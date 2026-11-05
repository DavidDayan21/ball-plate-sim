"""Session du bac à sable, sans affichage : actions, fin des tentatives, fichiers écrits."""
import csv
import math

import pytest

from simulation.session import Session


def advance(s: Session, duration: float) -> None:
    for _ in range(round(duration / s.dt)):
        s.step()


@pytest.fixture
def session(tmp_path):
    s = Session(folder=tmp_path)
    advance(s, 1.0)
    return s


def test_dropped_ball_bounces_then_stabilizes(session):
    session.throw(0.0, 0.0, height=0.305)                   # lâcher de 30,5 cm au centre
    advance(session, 15.0)
    l = session.summary[-1]
    assert l["action"] == "drop" and l["result"] == "stabilized"
    assert l["landings"] >= 5
    # Premier rebond : un peu sous e² · h = 0,88² × 305 mm (le plateau, monté sur moteurs, amortit)
    assert 150 < l["max_bounce_mm"] < 0.91 ** 2 * 305


def test_throw_caught_then_too_fast_falls(session):
    session.throw(0.10, 180, speed=0.4)
    advance(session, 15.0)
    assert session.summary[-1]["result"] == "stabilized"
    session.throw(0.10, 180, speed=1.0)
    advance(session, 3.0)
    assert session.summary[-1]["result"] == "fallen" and session.state == "fallen"
    advance(session, 2.0)                                     # la simulation continue sans balle
    session.center_ball()                                         # et repart
    advance(session, 5.0)
    assert session.summary[-1]["action"] == "center" and session.summary[-1]["result"] == "stabilized"


def test_disturbance_detected_automatically(session):
    session.center_ball()
    advance(session, 5.0)
    assert session.state == "waiting"
    session.phys.push(0.3, 0.0)                            # comme une poussée à la souris
    advance(session, 1.0)
    assert session.state == "attempt" and session.attempt["action"] == "disturbance"
    advance(session, 10.0)
    assert session.summary[-1]["action"] == "disturbance"
    assert session.summary[-1]["result"] == "stabilized"


def test_files_written(session):
    session.throw(0.10, 180, speed=0.4)
    advance(session, 2.0)
    session.set_setpoint("circle")
    session.free_attempt()                                 # interrompt le lancer
    advance(session, 2.0)
    summary = session.close()
    assert [l["result"] for l in summary] == ["interrupted", "interrupted"]

    with open(session.folder / "summary.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert len(rows) == 2 and rows[0]["action"] == "throw" and rows[1]["setpoint"] == "circle"
    assert rows[0]["speed_m_s"] == "0,400"                # virgule décimale (Excel en français)

    with open(session.folder / "attempt_002.csv", encoding="utf-8-sig") as f:
        measurements = list(csv.DictReader(f, delimiter=";"))
    number = lambda text: float(text.replace(",", "."))
    assert len(measurements) == 100                                # 2 s à 50 Hz
    # La consigne en cercle commence à son point de départ : 6 cm à 45°
    x, y = number(measurements[0]["x_setpoint_mm"]), number(measurements[0]["y_setpoint_mm"])
    assert math.isclose(x, 60 * math.cos(math.pi / 4), abs_tol=0.01) and math.isclose(y, x, abs_tol=0.01)
