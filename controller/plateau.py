"""Contrôleur du plateau 2D : un PID par axe + anticipation (feedforward)
+ compensation de la latence de mesure par prédiction (phase 6, désactivée par défaut).

Indépendant du simulateur : il reçoit la consigne, la mesure et l'accélération voulue,
et renvoie les deux inclinaisons (θx, θy) à appliquer au plateau.
"""
from collections import deque

from controller.pid import PID


class PredicteurRetard:
    """Reconstruit la position ACTUELLE de la balle à partir d'une mesure en retard de N périodes.

    1. Observateur α-β avec modèle : on estime (x, v) à l'instant de la mesure, en prédisant
       avec le modèle ẍ = c·g·θ et la commande appliquée à ce moment-là, puis en corrigeant
       par l'écart avec la mesure : x += α·e, v += (β/T)·e.
    2. Projection : on intègre le modèle sur les N périodes de retard, avec les N dernières
       commandes, que le contrôleur connaît puisque c'est lui qui les a envoyées.
    """

    def __init__(self, n_retard: int, periode: float, gain_modele: float, alpha: float = 0.5):
        self.n, self.T, self.g = n_retard, periode, gain_modele
        self.alpha = alpha
        self.beta = alpha ** 2 / (2 - alpha)          # réglage classique d'un filtre α-β
        self.commandes = deque([0.0] * (n_retard + 1), maxlen=n_retard + 1)  # u_{k-N-1} ... u_{k-1}
        self.x = self.v = None

    def predire(self, mesure: float) -> float:
        if self.x is None:                            # première mesure : on part de là, au repos
            self.x, self.v = mesure, 0.0
        else:
            # Prédiction de l'observateur jusqu'à l'instant de la mesure (avec u_{k-N-1})
            self.v += self.g * self.commandes[0] * self.T
            self.x += self.v * self.T
            # Correction par la mesure
            e = mesure - self.x
            self.x += self.alpha * e
            self.v += self.beta / self.T * e
        # Projection jusqu'à maintenant avec u_{k-N} ... u_{k-1}
        x, v = self.x, self.v
        for u in list(self.commandes)[1:]:
            v += self.g * u * self.T
            x += v * self.T
        return x

    def memoriser(self, commande: float) -> None:
        self.commandes.append(commande)


class ControleurPlateau:
    """Deux PID indépendants (axes x et y), saturés à ±theta_max.

    Anticipation : pour que la balle suive une trajectoire d'accélération a_ref, il faut
    θ_ff = a_ref / (c·g) (modèle linéarisé). Le PID n'a plus qu'à corriger les écarts.
    gain_modele = c·g = g / (1 + k), en (m/s²)/rad.

    compensation_retard = N > 0 : la mesure est supposée en retard de N périodes, et la
    position actuelle est reconstruite par PredicteurRetard avant d'entrer dans les PID.
    """

    def __init__(self, kp: float, ki: float, kd: float, theta_max: float,
                 tau_d: float, gain_modele: float, anticipation: bool = True,
                 compensation_retard: int = 0, periode: float = 0.02):
        self.pid_x = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.pid_y = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.gain_modele = gain_modele
        self.anticipation = anticipation
        self.predicteurs = ([PredicteurRetard(compensation_retard, periode, gain_modele) for _ in range(2)]
                            if compensation_retard > 0 else None)

    def calculer(self, consigne, mesure, acc_consigne, dt: float):
        """consigne, mesure, acc_consigne : couples (x, y). Renvoie (θx, θy) en rad."""
        if self.predicteurs is not None:
            mesure = (self.predicteurs[0].predire(mesure[0]), self.predicteurs[1].predire(mesure[1]))
        ffx = acc_consigne[0] / self.gain_modele if self.anticipation else 0.0
        ffy = acc_consigne[1] / self.gain_modele if self.anticipation else 0.0
        theta_x = self.pid_x.calculer(consigne[0], mesure[0], dt, ffx)
        theta_y = self.pid_y.calculer(consigne[1], mesure[1], dt, ffy)
        if self.predicteurs is not None:
            self.predicteurs[0].memoriser(theta_x)
            self.predicteurs[1].memoriser(theta_y)
        return theta_x, theta_y
