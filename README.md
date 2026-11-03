# ball-plate-sim

Simulation d'un robot équilibriste (« ball and plate ») : un plateau incliné par 3 moteurs maintient une balle en équilibre.
Objectif : valider les algorithmes de contrôle en simulation avant une éventuelle construction du robot.

## Avancement

- [x] Phase 0 — Environnement (Python, MuJoCo)
- [x] Phase 1 — Balle sur poutre en 1D (P, PD, PID)
- [x] Phase 2 — Balle sur plateau en 2D (bruit, retard, saturation, anticipation)
- [x] Phase 3 — Mécanisme à 3 bras et moteurs (cinématique inverse et directe, moteurs pas à pas)
- [x] Phase 4 — Simulation physique MuJoCo (chaîne fermée, contacts, rampes des moteurs)
- [x] Phase 5 — Capteur caméra (rendu MuJoCo, détection OpenCV, latence)
- [x] Phase 6 — Tests et robustesse (balle lancée, poussées, paramètres physiques, compensation de latence) — bilan : results/bilan_phase6.md
- [ ] Phase 7 — Robot réel (optionnel)

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_env.py
```

## Visualiser le robot dans MuJoCo

```powershell
python phase4_mujoco.py --vue              # simulation complète en temps réel
python -m sim_mujoco.modeles               # exporte les modèles XML
python -m mujoco.viewer --mjcf=sim_mujoco/robot.xml
```

## Tests

```powershell
python -m pytest
```

Chaque résultat théorique (erreur d'Euler, période, dépassement, erreur statique, critère de Routh, suivi de trajectoire, cinématique, physique MuJoCo, mesure par caméra, non-glissement, prédicteur de latence) est vérifié automatiquement.

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV.
