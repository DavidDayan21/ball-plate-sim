"""Phase 5 - Le robot est piloté uniquement à partir de l'image d'une caméra.

Caméra virtuelle fixe, 50 cm au-dessus du plateau, rendue par MuJoCo à chaque période de
contrôle (50 Hz). OpenCV détecte la balle (couleur orange), puis la géométrie de la caméra
convertit le pixel en position sur le plateau. Le reste de la chaîne est celui de la phase 4.

Variantes : capteur fictif de la phase 4, caméra 320×240, caméra 160×120, caméra sans
correction d'inclinaison, caméra avec 60 ms puis 100 ms de latence.

Lancement (depuis ball-plate-sim, environnement activé) :
    python phase5_camera.py         (quelques dizaines de secondes : 5 simulations avec rendu d'images)
"""
import math

import matplotlib.pyplot as plt

from sim_mujoco.boucle_mujoco import ConfigMujoco, simuler
from sim_mujoco.camera import ParametresCamera
from sim_simple.boucle_plateau import ecart_type_theta, erreur_suivi_rms, temps_reponse_5

variantes = {
    "Capteur fictif (phase 4)": ConfigMujoco(),
    "Caméra 320×240": ConfigMujoco(capteur="camera"),
    "Caméra 160×120": ConfigMujoco(capteur="camera", camera=ParametresCamera(largeur=160, hauteur_image=120)),
    "Caméra sans correction incl.": ConfigMujoco(capteur="camera", correction_inclinaison=False),
    "Caméra, latence 60 ms": ConfigMujoco(capteur="camera", retard=3),
    "Caméra, latence 100 ms": ConfigMujoco(capteur="camera", retard=5),
}

print(f"{'Variante':30s} {'Tombée':>6s} {'Retour 5 %':>10s} {'Err. cercle':>11s} {'Agit. θ':>8s} "
      f"{'Err. mesure':>11s} {'Non détectée':>12s}")
res = {}
for nom, cfg in variantes.items():
    r = simuler(cfg)
    res[nom] = r
    mesure = math.sqrt(sum(e * e for e in r.erreur_mesure) / len(r.erreur_mesure))
    print(f"{nom:30s} {'oui' if r.tombee else 'non':>6s} {temps_reponse_5(r, 4.0):>8.2f} s "
          f"{erreur_suivi_rms(r, 12, 20) * 1000:>8.2f} mm {math.degrees(ecart_type_theta(r, 2.5, 4)):>6.2f} ° "
          f"{mesure * 1000:>8.2f} mm {r.non_detectee:>12d}")
print("\nErr. mesure = écart quadratique moyen entre la position mesurée et la vraie position (avant latence).")
print("Latence = délai entre la prise de l'image et l'utilisation de la mesure par le contrôleur.")

# --- Figure 1 : ce que voit la caméra --------------------------------------------------
img, masque, centre = res["Caméra 320×240"].image_exemple
fig1, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.imshow(img)
a1.plot(centre[0] - 0.5, centre[1] - 0.5, "c+", markersize=14, markeredgewidth=2)
a1.set_title("Image de la caméra à t = 2 s (+ : centre détecté)")
a2.imshow(masque, cmap="gray")
a2.set_title("Masque OpenCV : pixels de couleur orange")
for a in (a1, a2):
    a.axis("off")
plt.tight_layout()

# --- Figure 2 : performance -------------------------------------------------------------
fig2, (b1, b2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
for nom in ("Capteur fictif (phase 4)", "Caméra 320×240", "Caméra, latence 100 ms"):
    r = res[nom]
    b1.plot(r.t, [math.hypot(a - c, b - d) * 1000 for a, b, c, d in zip(r.x, r.y, r.x_ref, r.y_ref)], label=nom)
b1.set_ylim(0, 40)
b1.set_ylabel("Erreur (mm)")
b1.set_title("Distance balle-consigne")
b1.legend(fontsize=8, loc="upper right")
b1.grid(True)

for nom in ("Capteur fictif (phase 4)", "Caméra 320×240"):
    r = res[nom]
    b2.plot(r.t, [e * 1000 for e in r.erreur_mesure], linewidth=0.8, label=nom)
b2.set_xlabel("Temps (s)")
b2.set_ylabel("Erreur de mesure (mm)")
b2.set_title("Précision du capteur : |position mesurée - vraie position|")
b2.legend(fontsize=8, loc="upper right")
b2.grid(True)

plt.tight_layout()
plt.show()
