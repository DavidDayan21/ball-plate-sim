"""Phase 6 - Tests de robustesse et bilan avant une éventuelle construction.

Chaîne complète et réaliste : MuJoCo (robot à 3 bras, rampes moteurs) + caméra + OpenCV.
La caméra est en 160×120 : aussi précise que 320×240 (phase 5), et plus rapide à simuler.

    1. Limites théoriques : condition de non-glissement, vitesse maximale rattrapable
    2. Balle lancée : vitesse maximale rattrapée (inclinaison max 10° puis 20°)
    3. Poussée sur une balle au repos au centre : choc maximal encaissé
    4. Robustesse aux paramètres physiques (scénario des phases 2 à 5)
    5. Latence de la caméra, avec et sans compensation par prédiction
    6. Trajectoire carrée

Lancement (depuis ball-plate-sim, environnement activé) :
    python phase6_robustesse.py     (1 à 3 min : une trentaine de simulations avec caméra)
Le bilan est aussi écrit dans results/bilan_phase6.md.
"""
import math
import os
from dataclasses import replace

import matplotlib.pyplot as plt

from sim_mujoco.boucle_mujoco import ConfigMujoco, acceleration_roulement, simuler
from sim_mujoco.camera import ParametresCamera
from sim_simple.boucle_plateau import erreur_suivi_rms, temps_reponse_5
from sim_simple.plateau import G

BASE = ConfigMujoco(capteur="camera", camera=ParametresCamera(largeur=160, hauteur_image=120))
PM = BASE.mujoco_params
K = BASE.plateau.k_inertie
CG = BASE.plateau.coefficient * G
R_PLATEAU = BASE.plateau.rayon_plateau
DEPART_LANCER = -0.10                 # la balle est lancée depuis x = -10 cm vers +x
md = ["# Bilan de la phase 6 : robustesse du robot équilibriste simulé\n",
      "Chaîne simulée : MuJoCo (robot à 3 bras, rampes moteurs) + caméra 160×120 + OpenCV + PID.\n"]


def section(titre: str) -> None:
    print(f"\n{titre}")
    md.append(f"\n## {titre}\n")


def tableau(entetes, lignes) -> None:
    largeurs = [max(len(str(x)) for x in col) for col in zip(entetes, *lignes)]
    for ligne in [entetes] + lignes:
        print("  " + "  ".join(str(x).rjust(l) if i else str(x).ljust(l) for i, (x, l) in enumerate(zip(ligne, largeurs))))
    md.append("| " + " | ".join(entetes) + " |")
    md.append("|" + "---|" * len(entetes))
    md.extend("| " + " | ".join(str(x) for x in ligne) + " |" for ligne in lignes)
    md.append("")


def centre(t: float):
    return 0.0, 0.0, 0.0, 0.0


def carre(t: float):
    """Carré de 10 cm de côté parcouru à 5 cm/s (2 s par côté), départ au coin (5 cm, 5 cm)."""
    coins = [(0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05), (0.05, -0.05)]
    k, s = int(t // 2.0) % 4, (t % 2.0) / 2.0
    (x0, y0), (x1, y1) = coins[k], coins[(k + 1) % 4]
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 0.0, 0.0


def v_max_theorique(theta_max: float, distance: float) -> float:
    """Freinage maximal a = c·g·sin θmax sur la distance disponible : v = √(2·a·d)."""
    return math.sqrt(2 * CG * math.sin(theta_max) * distance)


def balayage(cfg_de_v, vitesses):
    """Plus grande vitesse rattrapée : on augmente jusqu'à la première chute."""
    meilleure, resultats = None, {}
    for v in vitesses:
        r = simuler(cfg_de_v(v))
        resultats[v] = r
        if r.tombee:
            break
        meilleure = v
    return meilleure, resultats


# --- 1. Limites théoriques ----------------------------------------------------------------
section("1. Limites théoriques")
mu_10 = K * math.tan(math.radians(10)) / (1 + K)
a_roul = acceleration_roulement(K, math.radians(10), frottement=1.3 * mu_10)
a_gliss = acceleration_roulement(K, math.radians(10), frottement=0.7 * mu_10)
tableau(["Grandeur", "Théorie", "MuJoCo"], [
    ["Frottement minimal pour rouler à 10°", f"{mu_10:.3f}", "-"],
    ["Accélération à 10°, frottement 1,3×min (roule)", f"{G * math.sin(math.radians(10)) / (1 + K):.3f} m/s²", f"{a_roul:.3f} m/s²"],
    ["Accélération à 10°, frottement 0,7×min (glisse)",
     f"{G * (math.sin(math.radians(10)) - 0.7 * mu_10 * math.cos(math.radians(10))):.3f} m/s²", f"{a_gliss:.3f} m/s²"],
])

# --- 2. Balle lancée ----------------------------------------------------------------------
section("2. Balle lancée depuis x = -10 cm vers le bord opposé (consigne : centre)")
lignes, lancers = [], {}
for deg in (10, 20):
    th = math.radians(deg)
    vmax, res = balayage(lambda v: replace(BASE, theta_max=th, trajectoire=centre, duree=5.0,
                                           depart=(DEPART_LANCER, 0.0), vitesse_depart=(v, 0.0)),
                         [round(0.3 + 0.1 * i, 1) for i in range(8)])
    lancers[deg] = res
    lignes.append([f"±{deg}°", f"{vmax} m/s", f"{v_max_theorique(th, R_PLATEAU - DEPART_LANCER):.2f} m/s"])
tableau(["Inclinaison max", "Vitesse max rattrapée", "Limite physique √(2·c·g·sin θmax·d)"], lignes)

# --- 3. Poussée -----------------------------------------------------------------------------
section("3. Poussée sur la balle au repos au centre (t = 1 s)")
dvmax, _ = balayage(lambda dv: replace(BASE, trajectoire=centre, duree=5.0, depart=(0.0, 0.0),
                                       poussees=((1.0, 0.8 * dv, 0.6 * dv),)),
                    [round(0.2 + 0.1 * i, 1) for i in range(7)])
tableau(["Choc max encaissé (Δv)", "Limite physique (d = 15 cm, ±10°)"],
        [[f"{dvmax} m/s", f"{v_max_theorique(BASE.theta_max, R_PLATEAU):.2f} m/s"]])

# --- 4. Paramètres physiques -----------------------------------------------------------------
section("4. Robustesse aux paramètres physiques (contrôleur inchangé, réglé pour la balle de ping-pong)")
variantes = {
    "Nominal (ping-pong, μ = 1)": BASE,
    "Bille pleine (k = 2/5)": replace(BASE, mujoco_params=replace(PM, k_reel=2 / 5)),
    "Balle 10× plus lourde (27 g)": replace(BASE, mujoco_params=replace(PM, masse_balle=0.027)),
    "Frottement μ = 0,2": replace(BASE, mujoco_params=replace(PM, frottement=0.2)),
    "Frottement μ = 0,02": replace(BASE, mujoco_params=replace(PM, frottement=0.02)),
    "Plateau 2× plus lourd (0,5 kg)": replace(BASE, mujoco_params=replace(PM, masse_plateau=0.5)),
}
lignes, nominal, erreurs_param = [], None, {}
for nom, cfg in variantes.items():
    r = simuler(cfg)
    nominal = nominal or r
    erreurs_param[nom] = erreur_suivi_rms(r, 12, 20) * 1000
    lignes.append([nom, "oui" if r.tombee else "non", f"{temps_reponse_5(r, 4):.2f} s",
                   f"{erreur_suivi_rms(r, 12, 20) * 1000:.2f} mm", f"{100 * r.part_sans_contact:.1f} %"])
tableau(["Variante", "Tombée", "Retour 5 %", "Erreur cercle", "Sans contact"], lignes)

# --- 5. Latence et compensation --------------------------------------------------------------
section("5. Latence de la caméra : sans et avec compensation par prédiction")
lignes, latence = [], {}
for n in (1, 3, 5):
    for comp in (False, True):
        r = nominal if (n == 1 and not comp) else simuler(replace(BASE, retard=n, compensation_retard=comp))
        latence[(n, comp)] = r
        lignes.append([f"{20 * n} ms", "avec" if comp else "sans", "oui" if r.tombee else "non",
                       f"{temps_reponse_5(r, 4):.2f} s", f"{erreur_suivi_rms(r, 12, 20) * 1000:.2f} mm"])
tableau(["Latence", "Compensation", "Tombée", "Retour 5 %", "Erreur cercle"], lignes)

# --- 6. Carré -----------------------------------------------------------------------------------
section("6. Trajectoire carrée (10 cm de côté, 5 cm/s), latence 20 ms")
lignes, carres = [], {}
for comp in (False, True):
    r = simuler(replace(BASE, trajectoire=carre, duree=16.0, depart=(0.05, 0.05), compensation_retard=comp))
    carres[comp] = r
    lignes.append(["avec" if comp else "sans", "oui" if r.tombee else "non", f"{erreur_suivi_rms(r, 8, 16) * 1000:.2f} mm"])
tableau(["Compensation", "Tombée", "Erreur (2e tour)"], lignes)

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "bilan_phase6.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\nBilan écrit dans results/bilan_phase6.md")

# --- Figures -------------------------------------------------------------------------------------
fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 7.5))
for v, r in lancers[10].items():
    a1.plot(r.t, [x * 100 for x in r.x], label=f"{v} m/s" + (" (tombée)" if r.tombee else ""))
a1.axhline(R_PLATEAU * 100, color="red", linestyle="--", linewidth=0.8)
a1.set_xlim(0, 3)
a1.set_xlabel("Temps (s)")
a1.set_ylabel("x (cm)")
a1.set_title("Balle lancée, ±10° (rouge : bord du plateau)")
a1.legend(fontsize=7)
a1.grid(True)

for comp in (False, True):
    r = latence[(5, comp)]
    a2.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)],
            label="avec compensation" if comp else "sans compensation")
a2.set_ylim(0, 40)
a2.set_xlabel("Temps (s)")
a2.set_ylabel("Erreur (mm)")
a2.set_title("Latence 100 ms : distance balle-consigne")
a2.legend(fontsize=8)
a2.grid(True)

for comp, r in carres.items():
    garde = [i for i, t in enumerate(r.t) if t >= 8]
    a3.plot([r.x[i] * 100 for i in garde], [r.y[i] * 100 for i in garde],
            label="avec compensation" if comp else "sans compensation")
a3.plot([5, -5, -5, 5, 5], [5, 5, -5, -5, 5], "k--", linewidth=0.8, label="consigne")
a3.set_aspect("equal")
a3.set_xlabel("x (cm)")
a3.set_ylabel("y (cm)")
a3.set_title("Trajectoire carrée (2e tour)")
a3.legend(fontsize=8)
a3.grid(True)

noms = list(erreurs_param)
a4.barh(range(len(noms)), [erreurs_param[n] for n in noms], color="tab:blue")
a4.set_yticks(range(len(noms)))
a4.set_yticklabels(noms, fontsize=8)
a4.invert_yaxis()
a4.set_xlabel("Erreur sur le cercle (mm)")
a4.set_title("Robustesse aux paramètres physiques")
a4.grid(True, axis="x")
plt.tight_layout()
plt.show()
