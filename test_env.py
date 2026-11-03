"""Phase 0 - Vérification de l'environnement.

1. Affiche les versions des bibliothèques.
2. Simule la chute d'une balle avec MuJoCo pendant 1 seconde.
3. Trace la hauteur de la balle en fonction du temps (matplotlib).
4. Ouvre le visualiseur 3D de MuJoCo sur la même scène.
"""
import sys

import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# --- 1. Versions -------------------------------------------------------------
print("Python :", sys.version.split()[0])
print("numpy  :", np.__version__)
print("mujoco :", mujoco.__version__)
print("opencv :", cv2.__version__)

# --- 2. Scène : un sol + une balle de ping-pong lâchée à 1 m -----------------
XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="sol" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="balle" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

temps, hauteur = [], []
while data.time < 1.0:
    mujoco.mj_step(model, data)      # avance la simulation d'un pas (2 ms)
    temps.append(data.time)
    hauteur.append(data.qpos[2])     # qpos = [x, y, z, quaternion] -> z

print(f"Hauteur finale de la balle : {hauteur[-1]:.3f} m (attendu : environ 0.020 m)")

# --- 3. Graphique ------------------------------------------------------------
plt.plot(temps, hauteur)
plt.xlabel("Temps (s)")
plt.ylabel("Hauteur de la balle (m)")
plt.title("Phase 0 - Chute libre simulée par MuJoCo")
plt.grid(True)
print("Ferme la fenêtre du graphique pour ouvrir le visualiseur 3D...")
plt.show()

# --- 4. Visualiseur 3D -------------------------------------------------------
print("Visualiseur 3D ouvert (ferme la fenêtre pour terminer).")
mujoco.viewer.launch(model)
