"""Tests de la phase 1 : chaque résultat théorique démontré est vérifié automatiquement.

Lancement : python -m pytest
"""
import math
from dataclasses import replace

from controller.pid import PID
from sim_simple.poutre import (G, ParametresPoutre, acceleration, pas_euler,
                               pas_euler_semi_implicite, sur_la_poutre)

P = ParametresPoutre()
CG = P.coefficient * G


def boucle(pid, params=P, x0=0.10, dt=0.01, duree=10.0, pas=pas_euler_semi_implicite):
    """Boucle fermée 1D. Renvoie (temps, positions, tombée)."""
    x, v = x0, 0.0
    T, X = [], []
    for n in range(round(duree / dt) + 1):
        theta = pid.calculer(0.0, x, dt)
        T.append(n * dt)
        X.append(x)
        if not sur_la_poutre(x, params):
            return T, X, True
        x, v = pas(x, v, theta, dt, params)
    return T, X, False


def test_acceleration_balle_creuse():
    # ẍ = (3/5)·g·sin θ pour k = 2/3
    assert math.isclose(acceleration(math.radians(2), P), 0.6 * G * math.sin(math.radians(2)), rel_tol=1e-12)


def test_euler_explicite_erreur_ordre_1():
    # e_n = -½·a·t_n·dt (preuve de la phase 1a)
    theta = math.radians(2)
    a = acceleration(theta, P)
    for dt in (0.01, 0.005):
        x = v = 0.0
        n = round(1.0 / dt)
        for _ in range(n):
            x, v = pas_euler(x, v, theta, dt, P)
        assert math.isclose(x - 0.5 * a * 1.0 ** 2, -0.5 * a * 1.0 * dt, rel_tol=1e-6)


def test_correcteur_p_periode():
    # T0 = 2π/√(c·g·Kp) ; petite amplitude pour rester dans le domaine linéaire
    kp = 0.5
    T, X, tombee = boucle(PID(kp=kp), x0=0.01)
    passages = [T[i] for i in range(1, len(X)) if X[i - 1] * X[i] < 0]
    periode = 2 * (passages[-1] - passages[0]) / (len(passages) - 1)
    assert not tombee
    assert math.isclose(periode, 2 * math.pi / math.sqrt(CG * kp), rel_tol=0.01)


def test_correcteur_p_amplitude_conservee_semi_implicite():
    _, X, _ = boucle(PID(kp=1.0))
    assert max(abs(x) for x in X[-300:]) < 0.1005


def test_pd_depassement_zeta_07():
    zeta, kp = 0.7, 1.0
    kd = 2 * zeta * math.sqrt(CG * kp) / CG
    _, X, _ = boucle(PID(kp=kp, kd=kd))
    theorie = math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))
    assert abs(max(0.0, -min(X)) / 0.10 - theorie) < 0.005


def test_pd_erreur_statique_et_pid_qui_l_annule():
    biais = math.radians(0.5)
    p = replace(P, biais_theta=biais)
    kd = 2 * 0.7 * math.sqrt(CG) / CG
    _, X_pd, _ = boucle(PID(kp=1.0, kd=kd), params=p, duree=20)
    _, X_pid, _ = boucle(PID(kp=1.0, ki=1.0, kd=kd), params=p, duree=20)
    assert math.isclose(X_pd[-1], biais / 1.0, rel_tol=1e-3)      # e = θb / Kp
    assert abs(X_pid[-1]) < 1e-4


def test_pid_instable_au_dela_de_routh():
    kd = 2 * 0.7 * math.sqrt(CG) / CG
    limite = CG * 1.0 * kd                                           # Ki < c·g·Kp·Kd
    _, _, tombee_stable = boucle(PID(kp=1.0, ki=0.8 * limite, kd=kd), duree=30)
    _, _, tombee_instable = boucle(PID(kp=1.0, ki=1.2 * limite, kd=kd), duree=30)
    assert not tombee_stable
    assert tombee_instable
