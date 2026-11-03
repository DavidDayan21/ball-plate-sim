"""Contrôleur du plateau 2D : un PID par axe + anticipation (feedforward).

Indépendant du simulateur : il reçoit la consigne, la mesure et l'accélération voulue,
et renvoie les deux inclinaisons (θx, θy) à appliquer au plateau.
"""
from controller.pid import PID


class ControleurPlateau:
    """Deux PID indépendants (axes x et y), saturés à ±theta_max.

    Anticipation : pour que la balle suive une trajectoire d'accélération a_ref, il faut
    θ_ff = a_ref / (c·g) (modèle linéarisé). Le PID n'a plus qu'à corriger les écarts.
    gain_modele = c·g = g / (1 + k), en (m/s²)/rad.
    """

    def __init__(self, kp: float, ki: float, kd: float, theta_max: float,
                 tau_d: float, gain_modele: float, anticipation: bool = True):
        self.pid_x = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.pid_y = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.gain_modele = gain_modele
        self.anticipation = anticipation

    def calculer(self, consigne, mesure, acc_consigne, dt: float):
        """consigne, mesure, acc_consigne : couples (x, y). Renvoie (θx, θy) en rad."""
        ffx = acc_consigne[0] / self.gain_modele if self.anticipation else 0.0
        ffy = acc_consigne[1] / self.gain_modele if self.anticipation else 0.0
        theta_x = self.pid_x.calculer(consigne[0], mesure[0], dt, ffx)
        theta_y = self.pid_y.calculer(consigne[1], mesure[1], dt, ffy)
        return theta_x, theta_y
