"""Indicateurs de performance calculés sur un historique (simulation/boucle.py : Resultats)."""
import math


def indices(r, t0: float, t1: float):
    return [i for i, t in enumerate(r.t) if t0 <= t < t1]


def erreur_suivi_rms(r, t0: float, t1: float) -> float:
    """Écart quadratique moyen entre la balle et la consigne sur [t0, t1[ (m)."""
    idx = indices(r, t0, t1)
    return math.sqrt(sum((r.x[i] - r.x_ref[i]) ** 2 + (r.y[i] - r.y_ref[i]) ** 2 for i in idx) / len(idx))


def temps_reponse_5(r, t1: float) -> float:
    """Temps de réponse à 5 % de la distance initiale à la consigne (avant t1)."""
    d0 = math.hypot(r.x[0] - r.x_ref[0], r.y[0] - r.y_ref[0])
    for i in reversed(indices(r, 0.0, t1)):
        if math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) > 0.05 * d0:
            return r.t[i]
    return 0.0


def ecart_type_theta(r, t0: float, t1: float) -> float:
    """Écart-type de θx sur [t0, t1[ (rad) : agitation de la commande due au bruit."""
    vals = [r.theta_x[i] for i in indices(r, t0, t1)]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
