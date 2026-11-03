"""Phase 4 - Passage à MuJoCo : même contrôleur, vraie physique 3D.

1. Validation : MuJoCo retrouve-t-il l'accélération ẍ = g·sin θ / (1 + k) ?
2. Comparaison sur le scénario des phases 2 et 3 :
     - modèles maison (plateau idéal, robot)
     - MuJoCo : plateau commandé en angles (4a)
     - MuJoCo : robot complet, moteurs sans rampe puis avec rampes d'accélération (4b)

Lancement (depuis ball-plate-sim, environnement activé) :
    python phase4_mujoco.py          -> résultats + graphiques (environ 5 s de calcul)
    python phase4_mujoco.py --vue    -> le robot MuJoCo en 3D et en temps réel (20 s)
"""
import math
import sys

import matplotlib.pyplot as plt

from sim_mujoco.boucle_mujoco import ConfigMujoco, acceleration_roulement
from sim_mujoco.boucle_mujoco import simuler as simuler_mujoco
from sim_simple.boucle_plateau import Config, ecart_type_theta, erreur_suivi_rms, temps_reponse_5
from sim_simple.boucle_plateau import simuler as simuler_plateau
from sim_simple.boucle_robot import ConfigRobot
from sim_simple.boucle_robot import simuler as simuler_robot

if "--vue" in sys.argv:
    print("Robot MuJoCo avec rampes, en temps réel. Ferme la fenêtre pour arrêter.")
    r = simuler_mujoco(ConfigMujoco(), vue=True)
    print("La balle est tombée." if r.tombee else "Fin du scénario : la balle est restée sur le plateau.")
    sys.exit()


def couple_95(r) -> float:
    """Couple (N·m) que le moteur le plus sollicité ne dépasse pas 95 % du temps."""
    pics = sorted(max(abs(c) for c in cc) for cc in r.couples)
    return pics[int(0.95 * len(pics))]


# --- 1. Validation de la physique du roulement -------------------------------------------
print("1. MuJoCo et la théorie du roulement (plan fixe incliné de 2°)")
theta = math.radians(2)
for k, nom in ((2 / 3, "balle creuse (k = 2/3)"), (2 / 5, "bille pleine (k = 2/5)")):
    a = acceleration_roulement(k, theta)
    theorie = 9.81 * math.sin(theta) / (1 + k)
    print(f"  {nom:23s}: MuJoCo {a:.4f} m/s², théorie {theorie:.4f} m/s², écart {100 * (a / theorie - 1):+.2f} %")

# --- 2. Comparaison sur le scénario commun ------------------------------------------------
variantes = {
    "Maison : plateau idéal (ph. 2)": lambda: simuler_plateau(Config()),
    "Maison : robot (ph. 3)": lambda: simuler_robot(ConfigRobot()),
    "MuJoCo : plateau en angles": lambda: simuler_mujoco(ConfigMujoco(mecanisme=False)),
    "MuJoCo : robot sans rampe": lambda: simuler_mujoco(ConfigMujoco(acceleration_max=math.inf)),
    "MuJoCo : robot avec rampes": lambda: simuler_mujoco(ConfigMujoco()),
}
print("\n2. Même scénario que les phases 2 et 3")
print(f"{'Variante':31s} {'Tombée':>6s} {'Retour 5 %':>10s} {'Err. cercle':>11s} {'Agit. θ':>8s} "
      f"{'Sans contact':>12s} {'Couple 95 %':>12s}")
res = {}
for nom, lancer in variantes.items():
    r = lancer()
    res[nom] = r
    contact = f"{100 * r.part_sans_contact:10.1f} %" if hasattr(r, "part_sans_contact") else f"{'-':>12s}"
    couple = (f"{couple_95(r):8.3f} N·m" if getattr(r, "couples", None) and r.couples[0] else f"{'-':>12s}")
    print(f"{nom:31s} {'oui' if r.tombee else 'non':>6s} {temps_reponse_5(r, 4.0):>8.2f} s "
          f"{erreur_suivi_rms(r, 12, 20) * 1000:>8.2f} mm {math.degrees(ecart_type_theta(r, 2.5, 4)):>6.2f} ° "
          f"{contact} {couple}")
print("\nSans contact = part du temps où la balle a décollé du plateau.")
print(f"Couple 95 % = couple que le moteur le plus sollicité ne dépasse pas 95 % du temps "
      f"(limite simulée : {ConfigMujoco().mujoco_params.couple_max} N·m).")

# --- 3. Graphiques ------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for nom in ("Maison : robot (ph. 3)", "MuJoCo : robot sans rampe", "MuJoCo : robot avec rampes"):
    r = res[nom]
    ax1.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)], label=nom)
ax1.set_ylim(0, 30)
ax1.set_ylabel("Erreur (mm)")
ax1.set_title("Distance balle-consigne")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True)

FENETRE = 25                                           # moyenne glissante sur 0,5 s
for nom in ("MuJoCo : robot sans rampe", "MuJoCo : robot avec rampes"):
    r = res[nom]
    v = r.sans_contact
    lisse = [100 * sum(v[max(0, i - FENETRE + 1):i + 1]) / len(v[max(0, i - FENETRE + 1):i + 1]) for i in range(len(v))]
    ax2.plot(r.t, lisse, label=nom)
ax2.set_ylabel("Sans contact (%)")
ax2.set_title("Part du temps où la balle décolle du plateau (moyenne sur 0,5 s)")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True)

r = res["MuJoCo : robot avec rampes"]
for i in range(3):
    ax3.plot(r.t, [c[i] for c in r.couples], linewidth=0.8, label=f"Moteur {i + 1}")
cmax = ConfigMujoco().mujoco_params.couple_max
for s in (1, -1):
    ax3.axhline(s * cmax, color="red", linestyle="--", linewidth=0.7)
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Couple (N·m)")
ax3.set_title("Robot MuJoCo avec rampes : couples des moteurs (rouge : couple max)")
ax3.legend(fontsize=8, ncol=3, loc="lower left")
ax3.grid(True)

plt.tight_layout()
plt.show()
