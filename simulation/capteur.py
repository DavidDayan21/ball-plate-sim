"""Capteur fictif de position : bruit gaussien + retard (latence).

Sert aussi de file d'attente pour la latence de la caméra (simulation/camera.py).
"""
from collections import deque
import random


class Capteur:
    """Renvoie la position mesurée avec un bruit gaussien d'écart-type sigma (m),
    et un retard de `retard` périodes de contrôle."""

    def __init__(self, sigma: float = 0.001, retard: int = 1, graine: int = 0):
        self.sigma = sigma
        self.retard = retard
        self.rng = random.Random(graine)     # graine fixe : résultats reproductibles
        self.file = deque()

    def mesurer(self, x: float, y: float):
        """À appeler une fois par période de contrôle. Renvoie (x_mesuré, y_mesuré)."""
        self.file.append((x + self.rng.gauss(0, self.sigma), y + self.rng.gauss(0, self.sigma)))
        if len(self.file) > self.retard:
            return self.file.popleft()
        return self.file[0]                  # au tout début : pas encore d'historique
