"""Phase 1c - Correcteurs PD puis PID.

A. PD : θ = Kp·e + Kd·de/dt, avec Kp = 1 rad/m fixe.
   Kd est choisi pour viser un facteur d'amortissement ζ : Kd = 2·ζ·ω0 / (c·g).
   On compare le dépassement et le temps de réponse à 5 % mesurés à la solution
   exacte du modèle linéarisé.

B. PID : la poutre a un défaut d'étalonnage de 0,5° que le contrôleur ignore.
   Le PD laisse une erreur statique θb/Kp ; le terme intégral l'annule.
   Au-delà de Ki = c·g·Kp·Kd (critère de Routh), la boucle devient instable.

Lancement (depuis le dossier ball-plate-sim, environnement activé) :
    python phase1c_pd_pid.py
"""
import math
from dataclasses import replace

import matplotlib.pyplot as plt

from controller.pid import PID
from sim_simple.poutre import G, ParametresPoutre, pas_euler_semi_implicite, sur_la_poutre

p = ParametresPoutre()
CG = p.coefficient * G        # c·g = g/(1+k)
X0 = 0.10                     # position initiale (m)
CONSIGNE = 0.0                # position voulue (m)
DT = 0.01                     # période du contrôleur et pas de simulation (s)
KP = 1.0                      # gain proportionnel (rad/m), identique à la phase 1b
W0 = math.sqrt(CG * KP)       # pulsation propre (rad/s)


def simuler(pid: PID, params: ParametresPoutre, duree: float):
    """Boucle fermée : mesure -> contrôleur -> θ -> physique (Euler semi-implicite)."""
    x, v = X0, 0.0
    T, X, TH, TERMES = [], [], [], []
    for n in range(round(duree / DT) + 1):
        theta = pid.calculer(CONSIGNE, x, DT)
        T.append(n * DT)
        X.append(x)
        TH.append(theta)
        TERMES.append(pid.termes)                 # (P, I, D) à cet instant
        if not sur_la_poutre(x, params):
            break
        x, v = pas_euler_semi_implicite(x, v, theta, DT, params)
    return T, X, TH, TERMES


def kd_pour(zeta: float) -> float:
    """Gain dérivé donnant le facteur d'amortissement ζ : 2·ζ·ω0 = c·g·Kd."""
    return 2 * zeta * W0 / CG


def x_theorique(t: float, zeta: float) -> float:
    """Solution exacte de ẍ + 2ζω0·ẋ + ω0²·x = 0 avec x(0) = x0 et ẋ(0) = 0."""
    if zeta < 1:
        wd = W0 * math.sqrt(1 - zeta ** 2)
        return X0 * math.exp(-zeta * W0 * t) * (math.cos(wd * t) + zeta / math.sqrt(1 - zeta ** 2) * math.sin(wd * t))
    if zeta == 1:
        return X0 * math.exp(-W0 * t) * (1 + W0 * t)
    r1 = -W0 * (zeta - math.sqrt(zeta ** 2 - 1))
    r2 = -W0 * (zeta + math.sqrt(zeta ** 2 - 1))
    return X0 * (r2 * math.exp(r1 * t) - r1 * math.exp(r2 * t)) / (r2 - r1)


def depassement(X) -> float:
    """Dépassement de la consigne, en fraction de l'écart initial x0."""
    return max(0.0, -min(X)) / X0


def temps_reponse_5(T, X) -> float:
    """Temps de réponse à 5 % : dernier instant où |x| dépasse 5 % de x0."""
    for t, x in zip(reversed(T), reversed(X)):
        if abs(x) > 0.05 * X0:
            return t
    return 0.0


# --- A. Correcteur PD : effet de ζ ----------------------------------------------
DUREE_A = 10.0
print(f"A. Correcteur PD, Kp = {KP} rad/m (ω0 = {W0:.3f} rad/s)")
sim_A = {}
for zeta in (0.3, 0.7, 1.0, 2.0):
    kd = kd_pour(zeta)
    T, X, _, _ = simuler(PID(kp=KP, kd=kd), p, DUREE_A)
    sim_A[zeta] = (T, X)
    X_th = [x_theorique(t, zeta) for t in T]
    print(f"  ζ = {zeta:3.1f} (Kd = {kd:.3f} rad·s/m) : "
          f"dépassement = {depassement(X) * 100:5.1f} % (théorie {depassement(X_th) * 100:5.1f} %), "
          f"temps de réponse à 5 % = {temps_reponse_5(T, X):.2f} s (théorie {temps_reponse_5(T, X_th):.2f} s)")

# --- B. Défaut d'étalonnage : PD vs PID -----------------------------------------
DUREE_B = 20.0
BIAIS = math.radians(0.5)
p_biais = replace(p, biais_theta=BIAIS)
KD_B = kd_pour(0.7)
KI_LIMITE = CG * KP * KD_B
print(f"\nB. Poutre avec un défaut d'étalonnage de 0,5°, Kd = {KD_B:.3f} rad·s/m (ζ = 0,7)")
print(f"  Limite de stabilité (Routh) : Ki < c·g·Kp·Kd = {KI_LIMITE:.2f} rad/(m·s)")

cas_B = {
    "PD": PID(kp=KP, kd=KD_B),
    "PID, Ki = 1": PID(kp=KP, ki=1.0, kd=KD_B),
    f"PID, Ki = 4 (> {KI_LIMITE:.2f})": PID(kp=KP, ki=4.0, kd=KD_B),
}
sim_B = {}
for nom, pid in cas_B.items():
    T, X, TH, TERMES = simuler(pid, p_biais, DUREE_B)
    sim_B[nom] = (T, X, TERMES)
    if T[-1] < DUREE_B:
        print(f"  {nom:18s}: instable, la balle tombe à t = {T[-1]:.2f} s")
        continue
    tp, ti, _ = TERMES[-1]
    print(f"  {nom:18s}: position finale = {X[-1] * 100:+.3f} cm, θ final = {math.degrees(TH[-1]):+.3f}° "
          f"= terme P {math.degrees(tp):+.3f}° + terme I {math.degrees(ti):+.3f}°")
print(f"  Erreur statique prévue pour le PD : θb/Kp = {BIAIS / KP * 100:.3f} cm")

# --- Graphiques ------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

for zeta, (T, X) in sim_A.items():
    ax1.plot(T, [x * 100 for x in X], label=f"ζ = {zeta}")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_xlim(0, 6)
ax1.set_ylabel("x (cm)")
ax1.set_title("A. Correcteur PD (Kp = 1 rad/m) : effet de l'amortissement ζ")
ax1.legend(ncol=4, fontsize=8)
ax1.grid(True)

for nom, (T, X, _) in sim_B.items():
    ax2.plot(T, [x * 100 for x in X], label=nom)
ax2.axhline(BIAIS / KP * 100, color="gray", linestyle="--", linewidth=0.8, label="Erreur statique prévue θb/Kp")
ax2.set_ylim(-12, 12)
ax2.set_ylabel("x (cm)")
ax2.set_title("B. Défaut d'étalonnage de 0,5° : le terme intégral annule l'erreur statique")
ax2.legend(ncol=2, fontsize=8)
ax2.grid(True)

T, _, TERMES = sim_B["PD"]
ax3.plot(T, [math.degrees(tp) for tp, _, _ in TERMES], label="PD : terme P")
T, _, TERMES = sim_B["PID, Ki = 1"]
ax3.plot(T, [math.degrees(tp) for tp, _, _ in TERMES], label="PID : terme P")
ax3.plot(T, [math.degrees(ti) for _, ti, _ in TERMES], label="PID : terme I")
ax3.axhline(-math.degrees(BIAIS), color="gray", linestyle="--", linewidth=0.8, label="-θb (compensation du défaut)")
ax3.set_ylim(-3, 1)
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Contribution à θ (°)")
ax3.set_title("Qui compense le défaut ? Le terme P (donc une erreur de position) ou le terme I")
ax3.legend(ncol=4, fontsize=8)
ax3.grid(True)

plt.tight_layout()
plt.show()
