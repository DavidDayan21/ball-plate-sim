"""Contrôleur PID discret.

Il ne connaît pas le simulateur : il reçoit une consigne et une mesure, et renvoie une commande.
Le même code servira dans la simulation maison, dans MuJoCo, puis sur le vrai robot.
"""
import math


class PID:
    """u = Kp·e + Ki·∫e·dt + Kd·de/dt,   avec e = consigne - mesure.

    Phase 1b : seul Kp est utilisé (Ki = Kd = 0). Phase 1c : PD puis PID complet.
    La dérivée est calculée par différence finie sur l'erreur : (e_n - e_{n-1}) / dt.

    Unités pour la balle sur poutre : e en m, u = θ en rad, donc Kp en rad/m.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 sortie_min: float = -math.inf, sortie_max: float = math.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.sortie_min, self.sortie_max = sortie_min, sortie_max
        self.reset()

    def reset(self) -> None:
        """Remet à zéro la mémoire du contrôleur (intégrale et erreur précédente)."""
        self.integrale = 0.0
        self.erreur_precedente = None
        self.termes = (0.0, 0.0, 0.0)     # (terme P, terme I, terme D) du dernier calcul

    def calculer(self, consigne: float, mesure: float, dt: float) -> float:
        """Calcule la commande pour un pas de contrôle de durée dt."""
        e = consigne - mesure
        self.integrale += e * dt
        derivee = 0.0 if self.erreur_precedente is None else (e - self.erreur_precedente) / dt
        self.erreur_precedente = e
        # Contribution de chaque terme, conservée pour l'analyse et les graphiques
        self.termes = (self.kp * e, self.ki * self.integrale, self.kd * derivee)
        u = sum(self.termes)
        return min(max(u, self.sortie_min), self.sortie_max)
