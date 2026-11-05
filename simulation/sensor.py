"""Capteur fictif de position : bruit gaussien + retard (latence).

Sert aussi de file d'attente pour la latence de la caméra (simulation/camera.py).
"""
from collections import deque
import random


class Sensor:
    """Renvoie la position mesurée avec un bruit gaussien d'écart-type sigma (m),
    et un retard de `retard` périodes de contrôle."""

    def __init__(self, sigma: float = 0.001, latency: int = 1, seed: int = 0):
        self.sigma = sigma
        self.latency = latency
        self.rng = random.Random(seed)     # graine fixe : résultats reproductibles
        self.queue = deque()

    def measure(self, x: float, y: float):
        """À appeler une fois par période de contrôle. Renvoie (x_mesuré, y_mesuré)."""
        self.queue.append((x + self.rng.gauss(0, self.sigma), y + self.rng.gauss(0, self.sigma)))
        if len(self.queue) > self.latency:
            return self.queue.popleft()
        return self.queue[0]                  # au tout début : pas encore d'historique
