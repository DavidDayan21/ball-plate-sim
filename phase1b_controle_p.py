"""Phase 1b - Premier contrôleur : le correcteur proportionnel (P).

Scénario : balle lâchée sans vitesse à 10 cm du centre, consigne = centre (x = 0).
Loi de commande : θ = Kp·(consigne - x), recalculée toutes les 10 ms (100 Hz).

Le script montre deux choses :
  A. Avec Kp = 1 rad/m, Euler explicite fait croître l'oscillation artificiellement,
     alors qu'Euler semi-implicite reste fidèle à la théorie (oscillation d'amplitude constante).
  B. Quel que soit Kp, le correcteur P seul donne une oscillation qui ne s'amortit jamais :
     Kp ne change que sa période, T = 2π / √(g·Kp/(1+k)).

Lancement (depuis le dossier ball-plate-sim, environnement activé) :
    python phase1b_controle_p.py
"""
import math

import matplotlib.pyplot as plt

from controller.pid import PID
from sim_simple.poutre import G, ParametresPoutre, pas_euler, pas_euler_semi_implicite, sur_la_poutre

p = ParametresPoutre()
X0 = 0.10          # position initiale (m)
CONSIGNE = 0.0     # position voulue (m) : le centre
DT = 0.01          # pas de simulation = période du contrôleur (s)
DUREE = 10.0       # durée simulée (s)


def simuler(kp: float, pas):
    """Boucle fermée : mesure -> contrôleur -> commande θ -> physique (un pas)."""
    pid = PID(kp=kp)
    x, v = X0, 0.0
    T, X, TH = [], [], []
    for n in range(round(DUREE / DT) + 1):
        theta = pid.calculer(CONSIGNE, x, DT)      # le contrôleur lit x et décide θ
        T.append(n * DT)
        X.append(x)
        TH.append(theta)
        if not sur_la_poutre(x, p):
            print(f"  La balle est tombée à t = {n * DT:.2f} s")
            break
        x, v = pas(x, v, theta, DT, p)             # la physique avance de dt avec θ constant
    return T, X, TH


def omega0(kp: float) -> float:
    """Pulsation propre du modèle linéarisé : ω0 = √(g·Kp/(1+k))."""
    return math.sqrt(p.coefficient * G * kp)


def periode_mesuree(T, X) -> float:
    """Période estimée à partir des passages par zéro (interpolation linéaire)."""
    passages = []
    for i in range(1, len(X)):
        if X[i - 1] * X[i] < 0:
            passages.append(T[i - 1] + (T[i] - T[i - 1]) * X[i - 1] / (X[i - 1] - X[i]))
    if len(passages) < 3:
        return float("nan")
    return 2 * (passages[-1] - passages[0]) / (len(passages) - 1)


def dernier_pic(T, X, kp: float):
    """Instant et amplitude du plus grand |x| sur la dernière période simulée."""
    t_debut = T[-1] - 2 * math.pi / omega0(kp)
    return max(((t, abs(x)) for t, x in zip(T, X) if t >= t_debut), key=lambda c: c[1])


# --- A. Euler explicite vs semi-implicite (Kp = 1 rad/m) ------------------------
KP_A = 1.0
w = omega0(KP_A)
print(f"A. Kp = {KP_A} rad/m : ω0 = {w:.3f} rad/s, période théorique = {2 * math.pi / w:.3f} s")

sim_A = {}
for nom, pas in (("Euler explicite", pas_euler), ("Euler semi-implicite", pas_euler_semi_implicite)):
    T, X, TH = simuler(KP_A, pas)
    sim_A[nom] = (T, X, TH)
    t_pic, a_pic = dernier_pic(T, X, KP_A)
    print(f"  {nom:21s}: dernier pic à t = {t_pic:.2f} s, amplitude = {a_pic * 100:.2f} cm "
          f"(départ : {X0 * 100:.0f} cm)")

# Prévision pour Euler explicite : l'amplitude est multipliée par √(1 + ω0²·dt²) à chaque pas
t_pic, _ = dernier_pic(*sim_A["Euler explicite"][:2], KP_A)
facteur = (1 + (w * DT) ** 2) ** (t_pic / DT / 2)
print(f"  Prévision Euler explicite au même instant : x{facteur:.3f}, soit {X0 * facteur * 100:.2f} cm")

# --- B. Influence de Kp (Euler semi-implicite) ----------------------------------
print("\nB. Influence de Kp (Euler semi-implicite)")
sim_B = {}
for kp in (0.5, 1.0, 2.0):
    T, X, TH = simuler(kp, pas_euler_semi_implicite)
    sim_B[kp] = (T, X)
    print(f"  Kp = {kp:3.1f} rad/m : période mesurée = {periode_mesuree(T, X):.3f} s, "
          f"théorique = {2 * math.pi / omega0(kp):.3f} s, "
          f"θ max = {math.degrees(kp * X0):.1f}°, "
          f"amplitude finale = {dernier_pic(T, X, kp)[1] * 100:.2f} cm")

# --- Graphiques ------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

for nom, (T, X, _) in sim_A.items():
    ax1.plot(T, [x * 100 for x in X], label=nom)
T = sim_A["Euler semi-implicite"][0]
ax1.plot(T, [X0 * math.cos(w * t) * 100 for t in T], "k--", linewidth=0.8,
         label="Théorie linéarisée x0·cos(ω0·t)")
ax1.set_ylabel("Position x (cm)")
ax1.set_title(f"A. Correcteur P, Kp = {KP_A} rad/m : effet de la méthode d'intégration")
ax1.set_ylim(-16, 21)
ax1.legend(loc="upper center", ncol=3, fontsize=8)
ax1.grid(True)

for kp, (T, X) in sim_B.items():
    ax2.plot(T, [x * 100 for x in X], label=f"Kp = {kp} rad/m")
ax2.set_ylabel("Position x (cm)")
ax2.set_title("B. Influence de Kp : la période change, l'oscillation ne s'amortit jamais")
ax2.set_ylim(-13, 17)
ax2.legend(loc="upper center", ncol=3, fontsize=8)
ax2.grid(True)

T, _, TH = sim_A["Euler semi-implicite"]
ax3.plot(T, [math.degrees(th) for th in TH], color="tab:green")
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Commande θ (°)")
ax3.set_title(f"Commande envoyée par le contrôleur (Kp = {KP_A} rad/m)")
ax3.grid(True)

plt.tight_layout()
plt.show()
