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
                 output_min: float = -math.inf, output_max: float = math.inf,
                 tau_d: float = 0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.tau_d = tau_d
        self.reset()

    def reset(self) -> None:
        """Remet à zéro la mémoire du contrôleur."""
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0
        self.terms = (0.0, 0.0, 0.0)     # (terme P, terme I, terme D) du dernier calcul

    def _saturate(self, u: float) -> float:
        return min(max(u, self.output_min), self.output_max)

    def compute(self, setpoint: float, measurement: float, dt: float, feedforward: float = 0.0) -> float:
        """Calcule la commande pour un pas de contrôle de durée dt."""
        e = setpoint - measurement

        # Dérivée filtrée : d <- a·d + (1 - a)·d_brute, avec a = tau_d / (tau_d + dt)
        raw = 0.0 if self.previous_error is None else (e - self.previous_error) / dt
        a = self.tau_d / (self.tau_d + dt)
        self.derivative = a * self.derivative + (1 - a) * raw
        self.previous_error = e

        up = self.kp * e
        ud = self.kd * self.derivative
        candidate_integral = self.integral + e * dt
        u = up + self.ki * candidate_integral + ud + feedforward
        u_sat = self._saturate(u)

        # Anti-windup : saturé ET l'erreur pousse dans le sens de la saturation -> on gèle l'intégrale
        if u_sat != u and (u - u_sat) * e > 0:
            u_sat = self._saturate(up + self.ki * self.integral + ud + feedforward)
        else:
            self.integral = candidate_integral

        self.terms = (up, self.ki * self.integral, ud)
        return u_sat
