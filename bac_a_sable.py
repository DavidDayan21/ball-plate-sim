"""Bac à sable 3D (version de base) : le robot complet dans MuJoCo, en temps réel.

Le contrôleur maintient la balle au centre pendant 2 minutes. Pour la perturber :
double-clic sur la balle, puis Ctrl + clic droit glissé pour la pousser.

La version complète (lancers, rebonds, consignes au clavier, enregistrement des données de
chaque tentative) est la prochaine étape du projet.

Lancement (depuis ball-plate-sim, environnement activé) :
    python bac_a_sable.py
"""
from simulation.boucle import simuler
from simulation.config import Config, centre

print("Robot MuJoCo en temps réel. Ferme la fenêtre pour arrêter.")
r = simuler(Config(trajectoire=centre, depart=(0.0, 0.0), duree=120.0), vue=True)
print("La balle est tombée." if r.tombee else "Fin : la balle est restée sur le plateau.")
