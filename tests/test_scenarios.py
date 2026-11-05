"""Scénarios complets en boucle fermée, de la physique simple à la chaîne MuJoCo + caméra."""
import math
from dataclasses import replace

from simulation.loop import simuler
from simulation.config import Config, centre
from simulation.metrics import ecart_type_theta, erreur_suivi_rms

PLATEAU_SIMPLE = Config(physique="simple", mecanisme=False)
ROBOT_SIMPLE = Config(physique="simple", acceleration_max=math.inf)


def cercle(cfg) -> float:
    return erreur_suivi_rms(simuler(cfg), 12, 20)


# --- Physique simple --------------------------------------------------------------------------
def test_filtre_derivee_reduit_le_bruit():
    filtre = ecart_type_theta(simuler(PLATEAU_SIMPLE), 2.5, 4)
    brut = ecart_type_theta(simuler(replace(PLATEAU_SIMPLE, tau_d=0.0)), 2.5, 4)
    assert filtre < 0.5 * brut


def test_anticipation_ameliore_le_suivi():
    assert cercle(PLATEAU_SIMPLE) < 0.2 * cercle(replace(PLATEAU_SIMPLE, anticipation=False))


def test_plateau_simple():
    r = simuler(PLATEAU_SIMPLE)
    assert not r.tombee and erreur_suivi_rms(r, 12, 20) < 0.006


def test_robot_simple():
    r = simuler(ROBOT_SIMPLE)
    assert not r.tombee and r.poses_hors_portee == 0 and erreur_suivi_rms(r, 12, 20) < 0.006


def test_moteurs_lents_degradent_le_suivi():
    assert cercle(replace(ROBOT_SIMPLE, vitesse_max=0.5)) > 1.5 * cercle(ROBOT_SIMPLE)


# --- MuJoCo -------------------------------------------------------------------------------------
def test_mujoco_plateau_en_angles():
    r = simuler(Config(mecanisme=False))
    assert not r.tombee and erreur_suivi_rms(r, 12, 20) < 0.008


def test_mujoco_robot_avec_rampes():
    r = simuler(Config())
    assert not r.tombee and r.poses_hors_portee == 0
    assert erreur_suivi_rms(r, 12, 20) < 0.008
    assert r.part_decollee < 0.01                      # la balle ne décolle quasiment pas


def test_rampes_suppriment_les_rebonds():
    # Balle élastique (e ≈ 0,88) : sans rampes, les à-coups du plateau la font rebondir
    assert simuler(Config(acceleration_max=math.inf)).part_decollee > 0.5
    assert simuler(Config()).part_decollee < 0.01


def test_robot_pilote_par_la_camera():
    r = simuler(Config(capteur="camera"))
    assert not r.tombee and r.non_detectee == 0 and erreur_suivi_rms(r, 12, 20) < 0.008


def test_latence_degrade_le_suivi():
    assert cercle(Config(capteur="camera", retard=5)) > 2 * cercle(Config(capteur="camera"))


def test_compensation_de_latence():
    assert cercle(Config(retard=5, compensation_retard=True)) < 0.5 * cercle(Config(retard=5))


def test_balle_lancee_rattrapee():
    r = simuler(Config(trajectoire=centre, duree=5.0, depart=(-0.10, 0.0), vitesse_depart=(0.4, 0.0)))
    assert not r.tombee and math.hypot(r.x[-1], r.y[-1]) < 0.01


def test_la_masse_de_la_balle_ne_change_rien():
    # ẍ = g·sin θ / (1 + k) : la masse se simplifie
    pm = Config().mujoco_params
    lourde = cercle(Config(mujoco_params=replace(pm, masse_balle=0.027)))
    assert abs(lourde / cercle(Config()) - 1) < 0.15
