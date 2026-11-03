"""Tests de la phase 2 : plateau 2D, capteur, PID amélioré, boucle complète.

Lancement : python -m pytest
"""
import math
from dataclasses import replace

from controller.pid import PID
from sim_simple.boucle_plateau import Config, ecart_type_theta, erreur_suivi_rms, simuler
from sim_simple.capteur import Capteur
from sim_simple.plateau import ParametresPlateau, acceleration
from sim_simple.poutre import ParametresPoutre
from sim_simple.poutre import acceleration as acceleration_1d


def test_plateau_axes_decouples_et_identiques_a_la_poutre():
    p2, p1 = ParametresPlateau(), ParametresPoutre()
    ax, ay = acceleration(math.radians(3), 0.0, p2)
    assert math.isclose(ax, acceleration_1d(math.radians(3), p1), rel_tol=1e-12)
    assert ay == 0.0


def test_capteur_retard_exact():
    c = Capteur(sigma=0.0, retard=2)
    sorties = [c.mesurer(float(k), 0.0)[0] for k in range(6)]
    assert sorties == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_pid_par_defaut_inchange():
    # Sans filtre ni saturation : même résultat que le PID des phases 1b-1c
    pid = PID(kp=2.0, ki=0.5, kd=0.3)
    u1 = pid.calculer(1.0, 0.0, 0.1)          # e = 1, dérivée = 0, intégrale = 0,1
    u2 = pid.calculer(1.0, 0.5, 0.1)          # e = 0,5, dérivée = -5, intégrale = 0,15
    assert math.isclose(u1, 2.0 + 0.05, rel_tol=1e-12)
    assert math.isclose(u2, 1.0 + 0.075 - 1.5, rel_tol=1e-12)


def test_anti_windup_gele_l_integrale():
    pid = PID(kp=1.0, ki=1.0, sortie_min=-0.1, sortie_max=0.1)
    for _ in range(100):
        assert pid.calculer(1.0, 0.0, 0.01) == 0.1   # saturé en permanence
    assert pid.integrale == 0.0                      # l'intégrale n'a pas grimpé


def test_filtre_derivee_reduit_le_bruit():
    r_filtre = simuler(Config())
    r_brut = simuler(replace(Config(), tau_d=0.0))
    assert ecart_type_theta(r_filtre, 2.5, 4) < 0.5 * ecart_type_theta(r_brut, 2.5, 4)


def test_scenario_complet():
    r = simuler(Config())
    assert not r.tombee
    assert erreur_suivi_rms(r, 12, 20) < 0.006                     # < 6 mm sur le cercle


def test_anticipation_ameliore_le_suivi():
    avec = erreur_suivi_rms(simuler(Config()), 12, 20)
    sans = erreur_suivi_rms(simuler(replace(Config(), anticipation=False)), 12, 20)
    assert avec < 0.2 * sans
