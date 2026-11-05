"""Campagne de mesures : cohérence des modèles, robustesse, latence. Écrit results/report.md.

Chaîne de référence : MuJoCo (robot à 3 bras, rampes moteurs) + caméra 160×120 + OpenCV + PID.
La caméra 160×120 est aussi précise que 320×240, et plus rapide à simuler.

    0. Du modèle simple à la chaîne complète : ce que chaque niveau de réalisme ajoute
    1. Limites théoriques : condition de non-glissement ; rebond conforme à la règle ITTF
    2. Balle lancée : vitesse maximale rattrapée (inclinaison max 10° puis 20°)
    3. Poussée sur une balle au repos au centre : choc maximal encaissé
    4. Robustesse aux paramètres physiques
    5. Latence de la caméra, avec et sans compensation par prédiction
    6. Trajectoire carrée

Lancement (depuis ball-plate-sim, environnement activé) :
    python campaign.py        (quelques minutes : une trentaine de simulations)
"""
import math
import os
from dataclasses import replace

import matplotlib.pyplot as plt

from simulation.loop import simulate
from simulation.config import G, Config, CameraParams, square, center
from simulation.metrics import theta_std, tracking_error_rms, settling_time_5
from simulation.mujoco_physics import rolling_acceleration, measured_restitution

BASE = Config(sensor="camera", camera=CameraParams(width=160, image_height=120))
PM = BASE.mujoco_params
K = BASE.plate.k_inertia
CG = BASE.plate.coefficient * G
R_PLATE = BASE.plate.plate_radius
THROW_START = -0.10                 # la balle est lancée depuis x = -10 cm vers +x
md = ["# Bilan de la campagne de mesures du robot équilibriste simulé\n",
      "Chaîne simulée : MuJoCo (robot à 3 bras, rampes moteurs) + caméra 160×120 + OpenCV + PID.\n"]


def section(title: str) -> None:
    print(f"\n{title}")
    md.append(f"\n## {title}\n")


def table(headers, rows) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    for row in [headers] + rows:
        print("  " + "  ".join(str(x).rjust(l) if i else str(x).ljust(l) for i, (x, l) in enumerate(zip(row, widths))))
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---|" * len(headers))
    md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    md.append("")


def theoretical_v_max(theta_max: float, distance: float) -> float:
    """Freinage maximal a = c·g·sin θmax sur la distance disponible : v = √(2·a·d)."""
    return math.sqrt(2 * CG * math.sin(theta_max) * distance)


def sweep(cfg_of_v, speeds):
    """Plus grande vitesse rattrapée : on augmente jusqu'à la première chute."""
    best, results = None, {}
    for v in speeds:
        r = simulate(cfg_of_v(v))
        results[v] = r
        if r.fallen:
            break
        best = v
    return best, results


# --- 0. Du modèle simple à la chaîne complète -----------------------------------------------
section("0. Du modèle simple à la chaîne complète (scénario de référence, mêmes réglages)")
levels = {
    "Maison : plateau en angles": Config(physics="simple", mechanism=False),
    "Maison : robot 3 bras": Config(physics="simple"),
    "MuJoCo : plateau en angles": Config(mechanism=False),
    "MuJoCo : robot 3 bras": Config(),
    "MuJoCo : robot 3 bras + caméra": BASE,
}
rows = []
for name, cfg in levels.items():
    r = simulate(cfg)
    contact = f"{100 * r.airborne_fraction:.1f} %" if cfg.physics == "mujoco" else "-"
    rows.append([name, "oui" if r.fallen else "non", f"{settling_time_5(r, 4):.2f} s",
                   f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm",
                   f"{math.degrees(theta_std(r, 2.5, 4)):.2f}°", contact])
table(["Niveau", "Tombée", "Retour 5 %", "Erreur cercle", "Agitation θ", "Décollée"], rows)

# --- 1. Limites théoriques ----------------------------------------------------------------
section("1. Limites théoriques et étalonnage du contact")
mu_10 = K * math.tan(math.radians(10)) / (1 + K)
a_roll = rolling_acceleration(K, math.radians(10), friction=1.3 * mu_10)
a_slide = rolling_acceleration(K, math.radians(10), friction=0.7 * mu_10)
table(["Grandeur", "Théorie", "MuJoCo"], [
    ["Frottement minimal pour rouler à 10°", f"{mu_10:.3f}", "-"],
    ["Accélération à 10°, frottement 1,3×min (roule)", f"{G * math.sin(math.radians(10)) / (1 + K):.3f} m/s²", f"{a_roll:.3f} m/s²"],
    ["Accélération à 10°, frottement 0,7×min (glisse)",
     f"{G * (math.sin(math.radians(10)) - 0.7 * mu_10 * math.cos(math.radians(10))):.3f} m/s²", f"{a_slide:.3f} m/s²"],
    ["Restitution, lâcher de 30,5 cm sur plan fixe (règle ITTF)", "0,85 à 0,91", f"{measured_restitution():.3f}"],
])

# --- 2. Balle lancée ----------------------------------------------------------------------
section("2. Balle lancée depuis x = -10 cm vers le bord opposé (consigne : centre)")
rows, throws = [], {}
for deg in (10, 20):
    th = math.radians(deg)
    vmax, res = sweep(lambda v: replace(BASE, theta_max=th, trajectory=center, duration=5.0,
                                           start_position=(THROW_START, 0.0), start_velocity=(v, 0.0)),
                         [round(0.3 + 0.1 * i, 1) for i in range(8)])
    throws[deg] = res
    rows.append([f"±{deg}°", f"{vmax} m/s", f"{theoretical_v_max(th, R_PLATE - THROW_START):.2f} m/s"])
table(["Inclinaison max", "Vitesse max rattrapée", "Limite physique √(2·c·g·sin θmax·d)"], rows)

# --- 3. Poussée -----------------------------------------------------------------------------
section("3. Poussée sur la balle au repos au centre (t = 1 s)")
dvmax, _ = sweep(lambda dv: replace(BASE, trajectory=center, duration=5.0, start_position=(0.0, 0.0),
                                       pushes=((1.0, 0.8 * dv, 0.6 * dv),)),
                    [round(0.2 + 0.1 * i, 1) for i in range(7)])
table(["Choc max encaissé (Δv)", "Limite physique (d = 15 cm, ±10°)"],
        [[f"{dvmax} m/s", f"{theoretical_v_max(BASE.theta_max, R_PLATE):.2f} m/s"]])

# --- 4. Paramètres physiques -----------------------------------------------------------------
section("4. Robustesse aux paramètres physiques (contrôleur inchangé, réglé pour la balle de ping-pong)")
variants = {
    "Nominal (ping-pong, μ = 1)": BASE,
    "Bille pleine (k = 2/5)": replace(BASE, mujoco_params=replace(PM, k_actual=2 / 5)),
    "Balle 10× plus lourde (27 g)": replace(BASE, mujoco_params=replace(PM, ball_mass=0.027)),
    "Frottement μ = 0,2": replace(BASE, mujoco_params=replace(PM, friction=0.2)),
    "Frottement μ = 0,02": replace(BASE, mujoco_params=replace(PM, friction=0.02)),
    "Plateau 2× plus lourd (0,5 kg)": replace(BASE, mujoco_params=replace(PM, plate_mass=0.5)),
}
rows, nominal, param_errors = [], None, {}
for name, cfg in variants.items():
    r = simulate(cfg)
    nominal = nominal or r
    param_errors[name] = tracking_error_rms(r, 12, 20) * 1000
    rows.append([name, "oui" if r.fallen else "non", f"{settling_time_5(r, 4):.2f} s",
                   f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm", f"{100 * r.airborne_fraction:.1f} %"])
table(["Variante", "Tombée", "Retour 5 %", "Erreur cercle", "Décollée"], rows)

# --- 5. Latence et compensation --------------------------------------------------------------
section("5. Latence de la caméra : sans et avec compensation par prédiction")
rows, latency = [], {}
for n in (1, 3, 5):
    for comp in (False, True):
        r = nominal if (n == 1 and not comp) else simulate(replace(BASE, latency=n, latency_compensation=comp))
        latency[(n, comp)] = r
        rows.append([f"{20 * n} ms", "avec" if comp else "sans", "oui" if r.fallen else "non",
                       f"{settling_time_5(r, 4):.2f} s", f"{tracking_error_rms(r, 12, 20) * 1000:.2f} mm"])
table(["Latence", "Compensation", "Tombée", "Retour 5 %", "Erreur cercle"], rows)

# --- 6. Carré -----------------------------------------------------------------------------------
section("6. Trajectoire carrée (10 cm de côté, 5 cm/s), latence 20 ms")
rows, squares = [], {}
for comp in (False, True):
    r = simulate(replace(BASE, trajectory=square, duration=16.0, start_position=(0.05, 0.05), latency_compensation=comp))
    squares[comp] = r
    rows.append(["avec" if comp else "sans", "oui" if r.fallen else "non", f"{tracking_error_rms(r, 8, 16) * 1000:.2f} mm"])
table(["Compensation", "Tombée", "Erreur (2e tour)"], rows)

os.makedirs("results", exist_ok=True)
with open(os.path.join("results", "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\nBilan écrit dans results/report.md")

# --- Figures -------------------------------------------------------------------------------------
fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 7.5))
for v, r in throws[10].items():
    a1.plot(r.t, [x * 100 for x in r.x], label=f"{v} m/s" + (" (tombée)" if r.fallen else ""))
a1.axhline(R_PLATE * 100, color="red", linestyle="--", linewidth=0.8)
a1.set_xlim(0, 3)
a1.set_xlabel("Temps (s)")
a1.set_ylabel("x (cm)")
a1.set_title("Balle lancée, ±10° (rouge : bord du plateau)")
a1.legend(fontsize=7)
a1.grid(True)

for comp in (False, True):
    r = latency[(5, comp)]
    a2.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)],
            label="avec compensation" if comp else "sans compensation")
a2.set_ylim(0, 40)
a2.set_xlabel("Temps (s)")
a2.set_ylabel("Erreur (mm)")
a2.set_title("Latence 100 ms : distance balle-consigne")
a2.legend(fontsize=8)
a2.grid(True)

for comp, r in squares.items():
    keep = [i for i, t in enumerate(r.t) if t >= 8]
    a3.plot([r.x[i] * 100 for i in keep], [r.y[i] * 100 for i in keep],
            label="avec compensation" if comp else "sans compensation")
a3.plot([5, -5, -5, 5, 5], [5, 5, -5, -5, 5], "k--", linewidth=0.8, label="setpoint")
a3.set_aspect("equal")
a3.set_xlabel("x (cm)")
a3.set_ylabel("y (cm)")
a3.set_title("Trajectoire carrée (2e tour)")
a3.legend(fontsize=8)
a3.grid(True)

names = list(param_errors)
a4.barh(range(len(names)), [param_errors[n] for n in names], color="tab:blue")
a4.set_yticks(range(len(names)))
a4.set_yticklabels(names, fontsize=8)
a4.invert_yaxis()
a4.set_xlabel("Erreur sur le cercle (mm)")
a4.set_title("Robustesse aux paramètres physiques")
a4.grid(True, axis="x")
plt.tight_layout()
plt.show()
