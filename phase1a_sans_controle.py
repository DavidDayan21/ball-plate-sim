"""Phase 1a - Balle sur poutre SANS contrôle.

Objectif : valider le modèle physique et la méthode d'Euler avant d'ajouter un contrôleur.

Scénario : poutre inclinée d'un angle FIXE θ = 2°, balle lâchée au centre sans vitesse.
Comme θ est constant, l'accélération est constante et la solution exacte est connue :
    x(t) = ½·a·t²
On compare donc la simulation à cette solution, et on mesure l'erreur d'Euler
pour deux pas de temps (10 ms et 5 ms).

Lancement (depuis le dossier ball-plate-sim, environnement activé) :
    python phase1a_sans_controle.py
"""
import math

import matplotlib.pyplot as plt

from sim_simple.poutre import ParametresPoutre, acceleration, pas_euler, sur_la_poutre

p = ParametresPoutre()
THETA = math.radians(2.0)          # inclinaison fixe de la poutre
a = acceleration(THETA, p)         # accélération constante de la balle


def simuler(dt: float):
    """Simule la balle jusqu'à ce qu'elle quitte la poutre. Renvoie (temps, positions)."""
    n, x, v = 0, 0.0, 0.0
    temps, positions = [0.0], [0.0]
    while sur_la_poutre(x, p):
        x, v = pas_euler(x, v, THETA, dt, p)
        n += 1
        temps.append(n * dt)       # n·dt plutôt que t += dt : évite l'accumulation d'arrondis
        positions.append(x)
    return temps, positions


def x_exact(t: float) -> float:
    return 0.5 * a * t ** 2


# --- Résultats numériques ------------------------------------------------------
t_sortie_theorique = math.sqrt(2 * (p.longueur / 2) / a)
print(f"Accélération : a = {a:.4f} m/s²")
print(f"Sortie de la poutre (théorie) : t = {t_sortie_theorique:.3f} s")

resultats = {}
for dt in (0.010, 0.005):
    T, X = simuler(dt)
    resultats[dt] = (T, X)
    print(f"Sortie de la poutre (Euler, dt = {dt * 1000:.0f} ms) : t = {T[-1]:.3f} s")

# Erreur à t = 1 s pour les deux pas de temps : le rapport doit valoir 2 (méthode d'ordre 1)
erreurs_1s = {}
for dt, (T, X) in resultats.items():
    n = round(1.0 / dt)
    erreurs_1s[dt] = X[n] - x_exact(T[n])
    print(f"Erreur à t = 1 s (dt = {dt * 1000:.0f} ms) : {erreurs_1s[dt] * 1000:+.3f} mm "
          f"(théorie -½·a·t·dt = {-0.5 * a * 1.0 * dt * 1000:+.3f} mm)")
print(f"Rapport des erreurs (10 ms / 5 ms) : {erreurs_1s[0.010] / erreurs_1s[0.005]:.2f} (attendu : 2)")

# --- Graphiques ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

T, X = resultats[0.010]
ax1.plot(T, [x_exact(t) * 100 for t in T], label="Solution exacte ½·a·t²")
ax1.plot(T, [x * 100 for x in X], ".", markersize=4, label="Euler, dt = 10 ms")
ax1.axhline(p.longueur / 2 * 100, color="red", linestyle="--", label="Bout de la poutre")
ax1.set_ylabel("Position x (cm)")
ax1.set_title("Phase 1a - Balle sur poutre inclinée de 2°, sans contrôle")
ax1.legend()
ax1.grid(True)

for dt, (T, X) in resultats.items():
    ax2.plot(T, [(x - x_exact(t)) * 1000 for t, x in zip(T, X)],
             label=f"Erreur Euler, dt = {dt * 1000:.0f} ms")
    ax2.plot(T, [-0.5 * a * t * dt * 1000 for t in T], "k--", linewidth=0.8)
ax2.plot([], [], "k--", linewidth=0.8, label="Théorie : -½·a·t·dt")
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel("Erreur (mm)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
