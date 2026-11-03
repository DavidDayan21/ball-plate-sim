# ball-plate-sim

Simulation d'un robot équilibriste (« ball and plate ») : un plateau incliné par 3 moteurs maintient
une balle en équilibre, la rattrape quand on la lance, et lui fait suivre une trajectoire.
Objectif : maîtriser entièrement la simulation (chaque écart expliqué et chiffré) avant toute construction.

## État d'avancement

| Étape | Contenu | État |
|---|---|---|
| Modèles | Physique maison (équations démontrées) et MuJoCo (3D, contacts, chaîne fermée) | Fait |
| Contrôle | PID filtré, anti-windup, anticipation, compensation de latence par prédiction | Fait |
| Mécanisme | 3 bras, cinématique inverse exacte et directe, moteurs pas à pas avec rampes | Fait |
| Capteur | Caméra virtuelle, détection OpenCV, projection sténopé | Fait |
| Robustesse | Balle lancée, poussées, paramètres physiques, latence (`campagne.py`) | Fait |
| Bac à sable 3D | Lancers, rebonds, consignes au clavier, données de chaque tentative | **Prochaine étape** |
| Écarts | Expliquer et chiffrer chaque écart entre modèles et objectifs | À faire |
| Contrôleur optimisé | Réglage systématique, LQR, approche de la limite physique | À faire |
| Conception | Taille et forme du plateau, longueur et nombre de bras | À faire |
| Défauts réels | Caméra (bruit, flou), jeu mécanique, courbe couple-vitesse | À faire |

La construction n'est envisagée qu'une fois toutes ces étapes validées.

## Structure

```text
controller/              Ce qui tournerait sur le vrai robot (indépendant du simulateur)
  pid.py                 PID : filtre de dérivée, saturation, anti-windup, anticipation
  plateau.py             2 PID (axes x, y) + anticipation + prédicteur de latence
  cinematique.py         Géométrie, cinématique inverse (exacte) et directe (Newton)
simulation/
  config.py              Configuration unique, paramètres physiques, trajectoires de consigne
  boucle.py              Boucle fermée unique : capteur -> contrôleur -> moteurs -> physique
  physique_simple.py     Physique maison (référence théorique)
  physique_mujoco.py     Physique MuJoCo : génération des modèles 3D, interface, validation
  moteur.py              Moteur pas à pas : vitesse max, micro-pas, butées, rampes
  capteur.py             Capteur fictif : bruit + latence
  camera.py              Caméra virtuelle : rendu, détection OpenCV, projection
  indicateurs.py         Indicateurs : erreur de suivi, temps de réponse, agitation de la commande
tests/                   Tests automatiques, par thème (contrôleur, mécanique, capteurs, physique, scénarios)
campagne.py              Campagne de mesures complète -> results/bilan.md
bac_a_sable.py           Le robot en 3D et en temps réel (version de base)
```

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Utilisation

```powershell
python -m pytest              # tous les tests (environ 30 s)
python campagne.py            # campagne de mesures complète, écrit results/bilan.md (1 à 3 min)
python bac_a_sable.py         # le robot en 3D ; double-clic sur la balle + Ctrl + clic droit pour la pousser
python -m simulation.physique_mujoco                   # exporte les modèles dans results/
python -m mujoco.viewer --mjcf=results/robot.xml       # ouvre le modèle seul dans MuJoCo
```

Une simulation se décrit par une `Config` ; tout ce qui n'est pas précisé prend la valeur de référence :

```python
from simulation.boucle import simuler
from simulation.config import Config, centre

r = simuler(Config(capteur="camera", retard=3, compensation_retard=True))
r = simuler(Config(physique="simple", mecanisme=False))           # modèle théorique
r = simuler(Config(trajectoire=centre, depart=(-0.1, 0.0), vitesse_depart=(0.4, 0.0)))  # balle lancée
```

## Conventions

- Unités SI : mètres, secondes, radians (degrés seulement à l'affichage).
- (x, y) : position de la balle dans le repère du plateau, origine au centre.
- θx > 0 abaisse le côté +x du plateau (la balle accélère vers +x) ; idem pour θy.
- Rotation du plateau : R = Ry(θx)·Rx(-θy).

## Principales hypothèses (aucun modèle réel de référence)

| Élément | Valeur de référence |
|---|---|
| Plateau | Disque de 15 cm de rayon, 0,25 kg |
| Balle | Ping-pong : 2 cm de rayon, 2,7 g, creuse (J = 2/3·m·r²) |
| Mécanisme | 3 bras de 5 cm, biellettes de 12 cm, rotules à 10 cm du centre |
| Moteurs | 10 rad/s, 1/16 de pas, rampes à 50 rad/s², couple max 0,4 N·m |
| Capteur | Caméra 50 cm au-dessus du plateau, 50 Hz, latence 20 ms |
| Contrôleur | 50 Hz, Kp = 1 rad/m, Ki = 1 rad/(m·s), ζ = 0,7, inclinaison max ±10° |

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV, pytest.
