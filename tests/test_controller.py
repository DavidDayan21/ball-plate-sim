"""Le contrôleur face à la théorie : chaque résultat démontré est vérifié.

Les essais en boucle fermée utilisent une seule direction du plateau (y = 0) de la physique
simple, dont la solution se calcule à la main.
"""
import math
from dataclasses import replace

from controller.pid import PID
from controller.plate import PredicteurRetard
from simulation.config import G, ParametresPlateau
from simulation.simple_physics import acceleration

P = ParametresPlateau()
CG = P.coefficient * G


def boucle_1d(pid, p=P, x0=0.10, dt=0.01, duree=10.0):
    """Balle sur une seule direction du plateau, Euler semi-implicite. Renvoie (temps, positions, tombée)."""
    x, v = x0, 0.0
    T, X = [], []
    for n in range(round(duree / dt) + 1):
        theta = pid.calculer(0.0, x, dt)
        T.append(n * dt)
        X.append(x)
        if abs(x) > p.rayon_plateau:
            return T, X, True
        v += acceleration(theta, 0.0, p)[0] * dt
        x += v * dt
    return T, X, False


def kd_pour(zeta, kp=1.0):
    return 2 * zeta * math.sqrt(CG * kp) / CG


def test_pid_calcul_elementaire():
    pid = PID(kp=2.0, ki=0.5, kd=0.3)
    u1 = pid.calculer(1.0, 0.0, 0.1)          # e = 1, dérivée = 0, intégrale = 0,1
    u2 = pid.calculer(1.0, 0.5, 0.1)          # e = 0,5, dérivée = -5, intégrale = 0,15
    assert math.isclose(u1, 2.0 + 0.05, rel_tol=1e-12)
    assert math.isclose(u2, 1.0 + 0.075 - 1.5, rel_tol=1e-12)


def test_anti_windup_gele_l_integrale():
    pid = PID(kp=1.0, ki=1.0, sortie_min=-0.1, sortie_max=0.1)
    for _ in range(100):
        assert pid.calculer(1.0, 0.0, 0.01) == 0.1
    assert pid.integrale == 0.0


def test_correcteur_p_periode():
    # Oscillateur harmonique : T0 = 2π / √(c·g·Kp) (petite amplitude : domaine linéaire)
    kp = 0.5
    T, X, tombee = boucle_1d(PID(kp=kp), x0=0.01)
    passages = [T[i] for i in range(1, len(X)) if X[i - 1] * X[i] < 0]
    periode = 2 * (passages[-1] - passages[0]) / (len(passages) - 1)
    assert not tombee
    assert math.isclose(periode, 2 * math.pi / math.sqrt(CG * kp), rel_tol=0.01)


def test_euler_semi_implicite_conserve_l_amplitude():
    _, X, _ = boucle_1d(PID(kp=1.0))
    assert max(abs(x) for x in X[-300:]) < 0.1005


def test_pd_depassement():
    # D = exp(-π·ζ / √(1 - ζ²))
    zeta = 0.7
    _, X, _ = boucle_1d(PID(kp=1.0, kd=kd_pour(zeta)))
    assert abs(max(0.0, -min(X)) / 0.10 - math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))) < 0.005


def test_pd_erreur_statique_et_pid_qui_l_annule():
    biais = math.radians(0.5)
    p = replace(P, biais_x=biais)
    _, X_pd, _ = boucle_1d(PID(kp=1.0, kd=kd_pour(0.7)), p=p, duree=20)
    _, X_pid, _ = boucle_1d(PID(kp=1.0, ki=1.0, kd=kd_pour(0.7)), p=p, duree=20)
    assert math.isclose(X_pd[-1], biais / 1.0, rel_tol=1e-3)      # e = θb / Kp
    assert abs(X_pid[-1]) < 1e-4


def test_pid_instable_au_dela_de_routh():
    # Stable si et seulement si Ki < c·g·Kp·Kd
    kd = kd_pour(0.7)
    limite = CG * 1.0 * kd
    _, _, tombee_stable = boucle_1d(PID(kp=1.0, ki=0.8 * limite, kd=kd), x0=0.05, duree=30)
    _, _, tombee_instable = boucle_1d(PID(kp=1.0, ki=1.2 * limite, kd=kd), x0=0.05, duree=30)
    assert not tombee_stable
    assert tombee_instable


def test_predicteur_exact_si_le_modele_est_exact():
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
