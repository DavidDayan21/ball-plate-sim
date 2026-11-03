"""Phase 3 - Le contrôleur pilote les 3 moteurs du mécanisme.

Chaîne simulée : capteur -> PID -> cinématique inverse -> 3 moteurs (vitesse bornée, micro-pas)
-> cinématique directe -> inclinaison réelle du plateau -> physique de la balle.

Même scénario que la phase 2. On compare :
    - le plateau idéal de la phase 2 (angles appliqués directement) ;
    - le robot avec moteurs réalistes (10 rad/s, 1/16 de pas) ;
    - le robot avec moteurs à pas entiers (1,8°) ;
    - le robot avec des moteurs de plus en plus lents (dimensionnement de la vitesse).

Lancement (depuis ball-plate-sim, environnement activé) :
    python phase3_robot.py        (environ 10 s de calcul)
"""
import math
from dataclasses import replace

import matplotlib.pyplot as plt

from controller.cinematique import Geometrie, bout_bras, cinematique_inverse, points_plateau, rotation
from sim_simple.boucle_plateau import ecart_type_theta, erreur_suivi_rms, temps_reponse_5
from sim_simple.boucle_robot import ConfigRobot, simuler

g = Geometrie()
BUTEE = math.radians(60)


def pose_possible(theta_x: float, theta_y: float) -> bool:
    try:
        return all(abs(a) <= BUTEE for a in cinematique_inverse(g, theta_x, theta_y))
    except ValueError:
        return False


def inclinaison_max() -> float:
    """Plus petite inclinaison maximale atteignable, toutes directions confondues (bissection)."""
    pire = math.inf
    for psi_deg in range(0, 360, 5):
        psi = math.radians(psi_deg)
        lo, hi = 0.0, math.radians(80)
        for _ in range(40):
            mid = (lo + hi) / 2
            if pose_possible(mid * math.cos(psi), mid * math.sin(psi)):
                lo = mid
            else:
                hi = mid
        pire = min(pire, lo)
    return pire


# --- 1. Vérifications géométriques ------------------------------------------------
print("1. Géométrie")
print(f"  Bras l1 = {g.longueur_bras * 100:.0f} cm, biellette l2 = {g.longueur_biellette * 100:.0f} cm, "
      f"rotules à {g.rayon_attache * 100:.0f} cm du centre")
print("  Repos : angles moteurs = " + ", ".join(f"{math.degrees(a):+.3f}°" for a in cinematique_inverse(g, 0, 0)))
a10 = cinematique_inverse(g, math.radians(10), 0)
print("  θx = 10° : angles moteurs = " + ", ".join(f"{math.degrees(a):+.2f}°" for a in a10)
      + f"  (rapport de transmission prévu : r/l1 = {g.rayon_attache / g.longueur_bras:.0f})")
print(f"  Inclinaison max atteignable dans toutes les directions : {math.degrees(inclinaison_max()):.1f}° "
      f"(butées moteurs à ±{math.degrees(BUTEE):.0f}°)")

# --- 2. Simulations -------------------------------------------------------------------
base = ConfigRobot()
variantes = {
    "Plateau idéal (phase 2)": replace(base, moteurs=False),
    "Robot : 10 rad/s, 1/16 de pas": base,
    "Robot : pas entiers (1,8°)": replace(base, resolution=math.radians(1.8)),
    "Robot : 3 rad/s": replace(base, vitesse_max=3.0),
    "Robot : 1 rad/s": replace(base, vitesse_max=1.0),
    "Robot : 0,5 rad/s": replace(base, vitesse_max=0.5),
}
print("\n2. Simulations (même scénario que la phase 2)")
print(f"{'Variante':31s} {'Tombée':>7s} {'Retour 5 %':>11s} {'Erreur cercle':>14s} {'Agitation θ':>12s}")
res = {}
for nom, cfg in variantes.items():
    r = simuler(cfg)
    res[nom] = r
    print(f"{nom:31s} {'oui' if r.tombee else 'non':>7s} {temps_reponse_5(r, 4.0):>9.2f} s "
          f"{erreur_suivi_rms(r, 12, 20) * 1000:>11.2f} mm {math.degrees(ecart_type_theta(r, 2.5, 4)):>10.2f} °")

# --- 3. Figure 1 : le mécanisme en 3D --------------------------------------------------
fig = plt.figure(figsize=(11, 5))
for k, (tx, ty, titre) in enumerate(((0.0, 0.0, "Au repos"), (math.radians(10), 0.0, "θx = 10°"))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    alphas = cinematique_inverse(g, tx, ty)
    P = points_plateau(g, tx, ty, g.hauteur)
    for i in range(3):
        ux, uy = g.direction(i)
        B = (g.rayon_base * ux, g.rayon_base * uy, 0.0)
        A = bout_bras(g, i, alphas[i])
        ax.plot(*zip(B, A), color="tab:blue", linewidth=3, label="Bras" if i == 0 else None)
        ax.plot(*zip(A, P[i]), color="tab:orange", linewidth=2, label="Biellette" if i == 0 else None)
        ax.scatter(*B, color="black", s=20)
    bord = [rotation(tx, ty, (0.15 * math.cos(u), 0.15 * math.sin(u), 0.0))
            for u in [2 * math.pi * j / 60 for j in range(61)]]
    ax.plot([b[0] for b in bord], [b[1] for b in bord], [b[2] + g.hauteur for b in bord], color="gray")
    ax.plot(*zip(*(P + [P[0]])), color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(0, 0.2)
    ax.set_box_aspect((1, 1, 0.67))
    ax.set_title(f"{titre} : α = " + ", ".join(f"{math.degrees(a):+.0f}°" for a in alphas))
    ax.legend(fontsize=8)

# --- 4. Figure 2 : moteurs et performance ---------------------------------------------
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
r = res["Robot : 10 rad/s, 1/16 de pas"]
for i in range(3):
    ax1.plot(r.t, [math.degrees(a[i]) for a in r.alphas], label=f"Moteur {i + 1}")
ax1.set_ylabel("Angle (°)")
ax1.set_title("Robot 10 rad/s : angles des 3 moteurs")
ax1.legend(fontsize=8, ncol=3, loc="lower left")
ax1.grid(True)

ax2.plot(r.t, [math.degrees(v) for v in r.theta_x], label="θx commandé")
ax2.plot(r.t, [math.degrees(v) for v in r.theta_x_reel], label="θx réel (cinématique directe)")
ax2.set_ylabel("θx (°)")
ax2.set_title("Robot 10 rad/s : inclinaison commandée et inclinaison réelle")
ax2.legend(fontsize=8, ncol=2, loc="lower left")
ax2.grid(True)

for nom in ("Plateau idéal (phase 2)", "Robot : 10 rad/s, 1/16 de pas", "Robot : 0,5 rad/s"):
    rr = res[nom]
    ax3.plot(rr.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(rr.x, rr.y, rr.x_ref, rr.y_ref)],
             label=nom)
ax3.set_ylim(0, 30)
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Erreur (mm)")
ax3.set_title("Distance balle-consigne")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True)

plt.tight_layout()
plt.show()
