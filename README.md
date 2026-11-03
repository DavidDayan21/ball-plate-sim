# ball-plate-sim

Simulation d'un robot équilibriste (« ball and plate ») : un plateau incliné par 3 moteurs maintient une balle en équilibre.
Objectif : valider les algorithmes de contrôle en simulation avant une éventuelle construction du robot.

## Avancement

- [x] Phase 0 — Environnement (Python, MuJoCo)
- [x] Phase 1 — Balle sur poutre en 1D (P, PD, PID)
- [x] Phase 2 — Balle sur plateau en 2D (bruit, retard, saturation, anticipation)
- [x] Phase 3 — Mécanisme à 3 bras et moteurs (cinématique inverse et directe, moteurs pas à pas)
- [ ] Phase 4 — Simulation physique MuJoCo
- [ ] Phase 5 — Capteur caméra (OpenCV)
- [ ] Phase 6 — Tests et robustesse
- [ ] Phase 7 — Robot réel (optionnel)

## Installation (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_env.py
```

## Tests

```powershell
python -m pytest
```

Chaque résultat théorique (erreur d'Euler, période, dépassement, erreur statique, critère de Routh, suivi de trajectoire, cinématique) est vérifié automatiquement.

## Technologies

Python, NumPy, Matplotlib, MuJoCo, OpenCV.
