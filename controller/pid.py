"""Contrôleur PID discret.

Il ne connaît pas le simulateur : il reçoit une consigne et une mesure, et renvoie une commande.
Le même code servira dans la simulation maison, dans MuJoCo, puis sur le vrai robot.
"""
import math


class PID:
    """u = Kp·e + Ki·∫e·dt + Kd·de/dt + u_ff,   avec e = consigne - mesure.

    - Dérivée : différence finie sur l'erreur, (e_n - e_{n-1}) / dt, filtrée par un
      passe-bas du 1er ordre de constante de temps tau_d (tau_d = 0 : pas de filtre).
    - Saturation : la sortie est bornée à [sortie_min, sortie_max].
    - Anti-windup : quand la sortie est saturée et que l'erreur pousse encore plus loin
      dans la saturation, l'intégrale est gelée.
    - Anticipation (feedforward) : u_ff est ajoutée avant la saturation.

    Avec les valeurs par défaut (pas de filtre, pas de saturation, u_ff = 0), on retrouve
    exactement le PID des phases 1b et 1c.

    Unités pour la balle sur poutre : e en m, u = θ en rad, donc Kp en rad/m.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 sortie_min: float = -math.inf, sortie_max: float = math.inf,
                 tau_d: float = 0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.sortie_min, self.sortie_max = sortie_min, sortie_max
        self.tau_d = tau_d
        self.reset()

    def reset(self) -> None:
        """Remet à zéro la mémoire du contrôleur."""
        self.integrale = 0.0
        self.erreur_precedente = None
        self.derivee = 0.0
        self.termes = (0.0, 0.0, 0.0)     # (terme P, terme I, terme D) du dernier calcul

    def _saturer(self, u: float) -> float:
        return min(max(u, self.sortie_min), self.sortie_max)

    def calculer(self, consigne: float, mesure: float, dt: float, anticipation: float = 0.0) -> float:
        """Calcule la commande pour un pas de contrôle de durée dt."""
        e = consigne - mesure

        # Dérivée filtrée : d <- a·d + (1 - a)·d_brute, avec a = tau_d / (tau_d + dt)
        brute = 0.0 if self.erreur_precedente is None else (e - self.erreur_precedente) / dt
        a = self.tau_d / (self.tau_d + dt)
        self.derivee = a * self.derivee + (1 - a) * brute
        self.erreur_precedente = e

        up = self.kp * e
        ud = self.kd * self.derivee
        integrale_candidate = self.integrale + e * dt
        u = up + self.ki * integrale_candidate + ud + anticipation
        u_sat = self._saturer(u)

        # Anti-windup : saturé ET l'erreur pousse dans le sens de la saturation -> on gèle l'intégrale
        if u_sat != u and (u - u_sat) * e > 0:
            u_sat = self._saturer(up + self.ki * self.integrale + ud + anticipation)
        else:
            self.integrale = integrale_candidate

        self.termes = (up, self.ki * self.integrale, ud)
        return u_sat
