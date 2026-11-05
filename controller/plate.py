"""Contrôleur du plateau 2D : un PID par axe + anticipation (feedforward)
+ compensation de la latence de mesure par prédiction (phase 6, désactivée par défaut).

Indépendant du simulateur : il reçoit la consigne, la mesure et l'accélération voulue,
et renvoie les deux inclinaisons (θx, θy) à appliquer au plateau.
"""
from collections import deque

from controller.pid import PID


class LatencyPredictor:
    """Reconstruit la position ACTUELLE de la balle à partir d'une mesure en retard de N périodes.

    1. Observateur α-β avec modèle : on estime (x, v) à l'instant de la mesure, en prédisant
       avec le modèle ẍ = c·g·θ et la commande appliquée à ce moment-là, puis en corrigeant
       par l'écart avec la mesure : x += α·e, v += (β/T)·e.
    2. Projection : on intègre le modèle sur les N périodes de retard, avec les N dernières
       commandes, que le contrôleur connaît puisque c'est lui qui les a envoyées.
    """

    def __init__(self, n_latency: int, period: float, model_gain: float, alpha: float = 0.5):
        self.n, self.T, self.g = n_latency, period, model_gain
        self.alpha = alpha
        self.beta = alpha ** 2 / (2 - alpha)          # réglage classique d'un filtre α-β
        self.commands = deque([0.0] * (n_latency + 1), maxlen=n_latency + 1)  # u_{k-N-1} ... u_{k-1}
        self.x = self.v = None

    def predict(self, measurement: float) -> float:
        if self.x is None:                            # première mesure : on part de là, au repos
            self.x, self.v = measurement, 0.0
        else:
            # Prédiction de l'observateur jusqu'à l'instant de la mesure (avec u_{k-N-1})
            self.v += self.g * self.commands[0] * self.T
            self.x += self.v * self.T
            # Correction par la mesure
            e = measurement - self.x
            self.x += self.alpha * e
            self.v += self.beta / self.T * e
        # Projection jusqu'à maintenant avec u_{k-N} ... u_{k-1}
        x, v = self.x, self.v
        for u in list(self.commands)[1:]:
            v += self.g * u * self.T
            x += v * self.T
        return x

    def store(self, command: float) -> None:
        self.commands.append(command)


class PlateController:
    """Deux PID indépendants (axes x et y), saturés à ±theta_max.

    Anticipation : pour que la balle suive une trajectoire d'accélération a_ref, il faut
    θ_ff = a_ref / (c·g) (modèle linéarisé). Le PID n'a plus qu'à corriger les écarts.
    gain_modele = c·g = g / (1 + k), en (m/s²)/rad.

    compensation_retard = N > 0 : la mesure est supposée en retard de N périodes, et la
    position actuelle est reconstruite par PredicteurRetard avant d'entrer dans les PID.
    """

    def __init__(self, kp: float, ki: float, kd: float, theta_max: float,
                 tau_d: float, model_gain: float, feedforward: bool = True,
                 latency_compensation: int = 0, period: float = 0.02):
        self.pid_x = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.pid_y = PID(kp, ki, kd, -theta_max, theta_max, tau_d)
        self.model_gain = model_gain
        self.feedforward = feedforward
        self.predictors = ([LatencyPredictor(latency_compensation, period, model_gain) for _ in range(2)]
                            if latency_compensation > 0 else None)

    def compute(self, setpoint, measurement, setpoint_acc, dt: float):
        """consigne, mesure, acc_consigne : couples (x, y). Renvoie (θx, θy) en rad."""
        if self.predictors is not None:
            measurement = (self.predictors[0].predict(measurement[0]), self.predictors[1].predict(measurement[1]))
        ffx = setpoint_acc[0] / self.model_gain if self.feedforward else 0.0
        ffy = setpoint_acc[1] / self.model_gain if self.feedforward else 0.0
        theta_x = self.pid_x.compute(setpoint[0], measurement[0], dt, ffx)
        theta_y = self.pid_y.compute(setpoint[1], measurement[1], dt, ffy)
        if self.predictors is not None:
            self.predictors[0].store(theta_x)
            self.predictors[1].store(theta_y)
        return theta_x, theta_y
