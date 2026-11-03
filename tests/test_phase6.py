"""Tests de la phase 6 : non-glissement, prédicteur de latence, robustesse.

Les scénarios utilisent le capteur fictif (sans rendu d'image) pour rester rapides.
Lancement : python -m pytest
"""
import math
from dataclasses import replace

from controller.plateau import PredicteurRetard
from sim_mujoco.boucle_mujoco import ConfigMujoco, acceleration_roulement, simuler
from sim_simple.boucle_plateau import erreur_suivi_rms

K, G = 2 / 3, 9.81


def centre(t):
    return 0.0, 0.0, 0.0, 0.0


def test_condition_de_non_glissement():
    # La balle roule si μ ≥ k·tan θ / (1 + k), sinon elle glisse : a = g·(sin θ - μ·cos θ)
    th = math.radians(10)
    mu_min = K * math.tan(th) / (1 + K)
    roule = acceleration_roulement(K, th, frottement=1.3 * mu_min)
    glisse = acceleration_roulement(K, th, frottement=0.7 * mu_min)
    assert abs(roule / (G * math.sin(th) / (1 + K)) - 1) < 0.01
    assert abs(glisse / (G * (math.sin(th) - 0.7 * mu_min * math.cos(th))) - 1) < 0.02


def test_predicteur_exact_si_le_modele_est_exact():
    # Système ẍ = g·u discrétisé comme dans le prédicteur, mesure en retard de N périodes
    T, g, N = 0.02, 5.886, 3
    p = PredicteurRetard(N, T, g)
    x, v, historique = 0.05, 0.1, []
    for k in range(300):
        historique.append(x)
        estimation = p.predire(historique[max(0, k - N)])
        u = 0.05 * math.sin(0.3 * k)
        p.memoriser(u)
        v += g * u * T
        x += v * T
    assert abs(estimation - historique[-1]) < 1e-9


def test_compensation_de_latence():
    sans = erreur_suivi_rms(simuler(replace(ConfigMujoco(), retard=5)), 12, 20)
    avec = erreur_suivi_rms(simuler(replace(ConfigMujoco(), retard=5, compensation_retard=True)), 12, 20)
    assert avec < 0.5 * sans


def test_balle_lancee_rattrapee():
    r = simuler(replace(ConfigMujoco(), trajectoire=centre, duree=5.0, depart=(-0.10, 0.0),
                        vitesse_depart=(0.4, 0.0)))
    assert not r.tombee
    assert math.hypot(r.x[-1], r.y[-1]) < 0.01          # revenue à moins de 1 cm du centre


def test_la_masse_de_la_balle_ne_change_rien():
    # Résultat de la phase 1 : la masse se simplifie dans ẍ = g·sin θ / (1 + k)
    pm = ConfigMujoco().mujoco_params
    nominal = erreur_suivi_rms(simuler(ConfigMujoco()), 12, 20)
    lourde = erreur_suivi_rms(simuler(replace(ConfigMujoco(), mujoco_params=replace(pm, masse_balle=0.027))), 12, 20)
    assert abs(lourde / nominal - 1) < 0.15
