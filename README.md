# ball-plate-sim

Simulation d'un robot équilibriste (« ball and plate ») : un plateau incliné par 3 moteurs maintient
une balle en équilibre, la rattrape quand on la lance, et lui fait suivre une trajectoire.
Objectif : un logiciel ludique, utilisable sans aucune notion de code, pour jouer avec le robot et
comparer deux « cerveaux » qui le pilotent : un PID et une IA locale. La physique est calculée par MuJoCo.
Construire le vrai robot reste un objectif lointain : le code sera alors prêt.

## État d'avancement

| Étape | Contenu | État |
|---|---|---|
| Modèles | Physique maison (équations démontrées) et MuJoCo (3D, contacts, chaîne fermée) | Fait |
| Contrôle | PID filtré, anti-windup, anticipation, compensation de latence par prédiction | Fait |
| Mécanisme | 3 bras, cinématique inverse exacte et directe, moteurs pas à pas avec rampes | Fait |
| Capteur | Caméra virtuelle, détection OpenCV, projection sténopé | Fait |
| Robustesse | Balle lancée, poussées, paramètres physiques, latence (`campaign.py`) | Fait |
| Contact réaliste | Rebond de balle de ping-pong conforme à la règle ITTF (e = 0,874) | Fait |
| Bac à sable 3D | Lancers, lâchers, rebonds, poussées, consignes, données de chaque tentative | Fait |
| A. Mesures fiables | Non-régression au bit près, dispersion sur N tirages, indicateurs robustes, écart maison → MuJoCo | **En cours** |
| B. Cerveau PID | Gains réglés par LQR, plateau à plat pendant les rebonds, commande adaptative | À faire |
| C. Cerveau IA locale | Entraînement dans MuJoCo, mêmes règles que le PID (capteur, moteurs, ±10°) | À faire |
| D. Logiciel grand public | Physique MuJoCo dans notre interface, choix du cerveau, comparatif | À faire |
| E. Construction (lointaine) | Conception, défauts réels, budget d'erreur complet | Plus tard |

Objectifs chiffrés des deux cerveaux : balle lancée rattrapée jusqu'à 90 % de la limite physique au moins,
erreur sur le cercle < 2 mm avec 60 ms de latence, retour au centre < 2 s, aucune chute dans les variantes
de robustesse.

## Structure

```text
controller/              Ce qui tournerait sur le vrai robot (indépendant du simulateur)
  pid.py                 PID : filtre de dérivée, saturation, anti-windup, anticipation
  plate.py               2 PID (axes x, y) + anticipation + prédicteur de latence
  kinematics.py          Géométrie, cinématique inverse (exacte) et directe (Newton)
simulation/
  config.py              Configuration unique, paramètres physiques, trajectoires de consigne
  loop.py                Boucle fermée unique : capteur -> contrôleur -> moteurs -> physique
  simple_physics.py      Physique maison (référence théorique)
  mujoco_physics.py      Physique MuJoCo : génération des modèles 3D, contact, interface, validation
  motor.py               Moteur pas à pas : vitesse max, micro-pas, butées, rampes
  sensor.py              Capteur fictif : bruit + latence
  camera.py              Caméra virtuelle : rendu, détection OpenCV, projection
  metrics.py             Indicateurs : erreur de suivi, temps de réponse, agitation de la commande
  session.py             Bac à sable : suite de tentatives, fin automatique, enregistrement CSV
  parallel.py            Simulations indépendantes réparties sur plusieurs processus
tests/                   Tests automatiques, par thème (contrôleur, mécanique, capteurs, physique,
                         scénarios, session, non-régression)
campaign.py              Campagne de mesures complète -> results/report.md
sandbox.py               Le robot en 3D et en temps réel, piloté depuis un panneau de commandes
regression_check.py      Vérifie qu'une modification laisse tous les résultats identiques au bit près
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
python -m pytest              # tous les tests
python campaign.py            # campagne de mesures complète, écrit results/report.md (quelques minutes)
python sandbox.py             # bac à sable 3D (voir ci-dessous)
python sandbox.py --camera                             # idem, balle mesurée par la caméra virtuelle
python -m simulation.mujoco_physics                    # exporte les modèles dans results/
python -m mujoco.viewer --mjcf=results/robot.xml       # ouvre le modèle seul dans MuJoCo
python regression_check.py --record       # avant une modification : référence de tous les résultats
python regression_check.py --compare      # après : vérifie qu'ils sont identiques au bit près
python regression_check.py --compare --by-position    # idem après un renommage (noms changés, même ordre)
```

Non-régression : toute refactorisation doit laisser les résultats identiques au bit près. La référence
(`results/reference/`, ignorée par git) couvre tous les cas de la campagne, la validation de la physique
et un enchaînement du bac à sable. Elle n'est valable que sur la machine et avec les versions qui l'ont
produite (Windows et Linux divergent dès que la balle rebondit) : la comparaison prévient si elles ont changé.

Une simulation se décrit par une `Config` ; tout ce qui n'est pas précisé prend la valeur de référence :

```python
from simulation.loop import simulate
from simulation.config import Config, center

r = simulate(Config(sensor="camera", latency=3, latency_compensation=True))
r = simulate(Config(physics="simple", mechanism=False))           # modèle théorique
r = simulate(Config(trajectory=center, start_position=(-0.1, 0.0), start_velocity=(0.4, 0.0)))  # balle lancée
```

## Bac à sable 3D

Deux fenêtres : la vue 3D MuJoCo et un panneau de commandes avec une vue de dessus et des courbes en direct.

| Commande | Effet |
|---|---|
| Lancer | Balle posée au point de départ (distance, angle), à la hauteur choisie, lancée vers le centre (déviation possible) |
| Lâcher | Même point de départ, même hauteur, sans vitesse : la balle tombe et rebondit |
| Centre | Balle posée immobile au centre |
| Tentative libre | Enregistre sans toucher à la balle |
| Clic dans la vue de dessus | Choisit le point de départ |
| Consigne, latence, compensation | Changent le comportement du contrôleur en direct |
| Barre d'espace (vue 3D) ou Pause | Pause / reprise |
| Double-clic sur la balle puis Ctrl + clic droit glissé (vue 3D) | Pousse la balle : une tentative « perturbation » démarre seule dès que la balle s'écarte de plus de 2 cm |

Une tentative se termine quand la balle tombe, reste 2 s à moins de 1 cm de la consigne (stabilisée),
dure plus de 30 s, ou quand une autre commence. Chaque tentative est enregistrée dans
`results/attempts/<date_heure>/` : `attempt_001.csv`… (toutes les mesures à 50 Hz) et `summary.csv`
(une ligne par tentative : paramètres, résultat, temps de stabilisation, rebonds, marge au bord…).
Format Excel français : séparateur `;`, virgule décimale.

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
| Contact balle-plateau | Restitution 0,88 visée (règle ITTF : lâchée de 30,5 cm, remonte de 22 à 25 cm), raideur 200 Hz, pas de simulation 0,5 ms |
| Mécanisme | 3 bras de 5 cm, biellettes de 12 cm, rotules à 10 cm du centre |
| Moteurs | 10 rad/s, 1/16 de pas, rampes à 50 rad/s², couple max 0,4 N·m |
| Capteur | Caméra 50 cm au-dessus du plateau, 50 Hz, latence 20 ms |
| Contrôleur | 50 Hz, Kp = 1 rad/m, Ki = 1 rad/(m·s), ζ = 0,7, inclinaison max ±10° |

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV, pytest.
