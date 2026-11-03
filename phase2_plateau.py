"""Phase 2 - Balle sur plateau 2D, avec les défauts du monde réel.

Défauts simulés : bruit de mesure (1 mm), retard de mesure (20 ms), contrôleur à 50 Hz,
saturation à ±10°, défauts d'étalonnage (0,5° et -0,3°).

Scénario (identique pour toutes les variantes) :
    0-4 s  : balle lâchée en (8 cm, -6 cm), retour au centre
    4-8 s  : consigne en (5 cm, 5 cm)
    8-20 s : suivi d'un cercle de 6 cm de rayon, un tour en 4 s

Variantes comparées :
    - Référence          : PID + filtre de dérivée + anti-windup + anticipation
    - Sans anticipation  : même chose, sans le terme θ_ff = a_ref / (c·g)
    - Sans filtre D      : même chose, sans filtrer la dérivée

Lancement (depuis ball-plate-sim, environnement activé) :
    python phase2_plateau.py           -> résultats + graphiques
    python phase2_plateau.py --anim    -> en plus, animation vue de dessus
"""
import math
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim_simple.boucle_plateau import (Config, ecart_type_theta, erreur_suivi_rms, kd_pour,
                                       simuler, temps_reponse_5)

base = Config()
variantes = {
    "Référence": base,
    "Sans anticipation": replace(base, anticipation=False),
    "Sans filtre D": replace(base, tau_d=0.0),
}
res = {nom: simuler(cfg) for nom, cfg in variantes.items()}

# --- Résultats ----------------------------------------------------------------------
print(f"Gains : Kp = {base.kp} rad/m, Ki = {base.ki} rad/(m·s), Kd = {kd_pour(base):.3f} rad·s/m (ζ = {base.zeta})")
print(f"Défauts : bruit {base.sigma * 1000:.0f} mm, retard {base.retard * base.periode_controle * 1000:.0f} ms, "
      f"contrôle à {1 / base.periode_controle:.0f} Hz, saturation ±{math.degrees(base.theta_max):.0f}°\n")
print(f"{'Variante':20s} {'Tombée':>7s} {'Retour 5 %':>11s} {'Erreur 7-8 s':>13s} "
      f"{'Erreur cercle':>14s} {'Agitation θ':>12s}")
for nom, r in res.items():
    print(f"{nom:20s} {'oui' if r.tombee else 'non':>7s} {temps_reponse_5(r, 4.0):>9.2f} s "
          f"{erreur_suivi_rms(r, 7, 8) * 1000:>10.2f} mm {erreur_suivi_rms(r, 12, 20) * 1000:>11.2f} mm "
          f"{math.degrees(ecart_type_theta(r, 2.5, 4)):>10.2f} °")
print("\nErreur cercle = écart quadratique moyen entre 12 et 20 s (2 tours).")
print("Agitation θ = écart-type de θx entre 2,5 et 4 s (balle au centre) : effet du bruit.")

# --- Graphiques -----------------------------------------------------------------------
r = res["Référence"]
fig = plt.figure(figsize=(11, 7))
grille = fig.add_gridspec(3, 2, width_ratios=[1, 1.4])

ax_vue = fig.add_subplot(grille[:, 0])
rayon = base.plateau.rayon_plateau * 100
ax_vue.add_patch(plt.Circle((0, 0), rayon, fill=False, color="gray", linestyle="--"))
ax_vue.plot([v * 100 for v in r.x_ref], [v * 100 for v in r.y_ref], "k--", linewidth=0.8, label="Consigne")
ax_vue.plot([v * 100 for v in r.x], [v * 100 for v in r.y], linewidth=1, label="Balle (référence)")
ax_vue.set_xlim(-rayon * 1.05, rayon * 1.05)
ax_vue.set_ylim(-rayon * 1.05, rayon * 1.05)
ax_vue.set_aspect("equal")
ax_vue.set_xlabel("x (cm)")
ax_vue.set_ylabel("y (cm)")
ax_vue.set_title("Vue de dessus (pointillés gris : bord du plateau)")
ax_vue.legend(fontsize=8, loc="lower left")
ax_vue.grid(True)

ax_pos = fig.add_subplot(grille[0, 1])
ax_pos.plot(r.t, [v * 100 for v in r.x], label="x")
ax_pos.plot(r.t, [v * 100 for v in r.y], label="y")
ax_pos.plot(r.t, [v * 100 for v in r.x_ref], "k--", linewidth=0.7, label="consignes")
ax_pos.plot(r.t, [v * 100 for v in r.y_ref], "k--", linewidth=0.7)
ax_pos.set_ylabel("Position (cm)")
ax_pos.set_title("Référence : position de la balle")
ax_pos.legend(fontsize=8, ncol=3, loc="lower left")
ax_pos.grid(True)

ax_th = fig.add_subplot(grille[1, 1], sharex=ax_pos)
ax_th.plot(r.t, [math.degrees(v) for v in r.theta_x], label="θx")
ax_th.plot(r.t, [math.degrees(v) for v in r.theta_y], label="θy")
for s in (1, -1):
    ax_th.axhline(s * math.degrees(base.theta_max), color="red", linestyle="--", linewidth=0.7)
ax_th.set_ylabel("Commande (°)")
ax_th.set_title("Référence : inclinaisons commandées (rouge : saturation)")
ax_th.legend(fontsize=8, ncol=2, loc="lower left")
ax_th.grid(True)

ax_err = fig.add_subplot(grille[2, 1], sharex=ax_pos)
for nom in ("Sans anticipation", "Référence"):
    rr = res[nom]
    ax_err.plot(rr.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(rr.x, rr.y, rr.x_ref, rr.y_ref)],
                label=nom)
ax_err.set_xlabel("Temps (s)")
ax_err.set_ylabel("Erreur (mm)")
ax_err.set_title("Distance balle-consigne : effet de l'anticipation")
ax_err.legend(fontsize=8, loc="upper left")
ax_err.grid(True)

plt.tight_layout()

# --- Animation (option --anim) -------------------------------------------------------
if "--anim" in sys.argv:
    fig_a, ax_a = plt.subplots(figsize=(6, 6))
    ax_a.add_patch(plt.Circle((0, 0), rayon, color="lightgray"))
    balle = plt.Circle((r.x[0] * 100, r.y[0] * 100), base.plateau.rayon_balle * 100, color="darkorange")
    ax_a.add_patch(balle)
    trace, = ax_a.plot([], [], color="darkorange", linewidth=1)
    cible, = ax_a.plot([], [], "kx", markersize=10)
    texte = ax_a.text(-rayon, rayon * 1.02, "")
    ax_a.set_xlim(-rayon * 1.1, rayon * 1.1)
    ax_a.set_ylim(-rayon * 1.1, rayon * 1.15)
    ax_a.set_aspect("equal")
    ax_a.set_title("Vue de dessus (x : consigne)")
    PAS = 2                                       # 1 image toutes les 2 mesures = 25 images/s

    def image(k):
        i = k * PAS
        balle.center = (r.x[i] * 100, r.y[i] * 100)
        debut = max(0, i - 50)                    # traînée : la dernière seconde
        trace.set_data([v * 100 for v in r.x[debut:i + 1]], [v * 100 for v in r.y[debut:i + 1]])
        cible.set_data([r.x_ref[i] * 100], [r.y_ref[i] * 100])
        texte.set_text(f"t = {r.t[i]:.1f} s")
        return balle, trace, cible, texte

    animation = FuncAnimation(fig_a, image, frames=len(r.t) // PAS, interval=40, blit=True, repeat=False)

plt.show()
