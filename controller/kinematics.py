"""Cinématique du mécanisme à 3 bras (phase 3).

Mécanisme : 3 moteurs placés à 120° autour du centre. Chaque moteur fait tourner un bras
de longueur l1 dans le plan vertical radial. Le bout du bras est relié au plateau par une
biellette de longueur l2, avec une rotule à chaque extrémité.

Repère : origine au niveau des axes moteurs, sur l'axe central ; z vers le haut.
Pose du plateau : centre en (0, 0, h), orientation donnée par les inclinaisons (θx, θy) du
contrôleur, avec la rotation R = Ry(θx)·Rx(-θy) : θx > 0 abaisse le côté +x, θy > 0 le côté +y.

Angle d'un bras α : 0 = bras horizontal pointant vers l'extérieur, > 0 vers le haut.

- cinematique_inverse : pose voulue -> angles des 3 moteurs (formule exacte). Tourne sur le robot.
- cinematique_directe : angles des moteurs -> pose réelle (méthode de Newton). Sert au simulateur.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    """Géométrie de départ (hypothèse, à recaler sur un modèle réel si on construit).

    Choisie pour qu'au repos le bras soit horizontal et la biellette verticale :
    rayon_attache = rayon_base + longueur_bras et hauteur = longueur_biellette.
    """
    base_radius: float = 0.05           # centre -> axe de chaque moteur (m)
    arm_length: float = 0.05        # l1 (m)
    rod_length: float = 0.12   # l2 (m)
    attachment_radius: float = 0.10        # centre du plateau -> rotule de la biellette (m)
    height: float = 0.12              # hauteur nominale du centre du plateau (m)
    arm_angles: tuple = (0.0, 120.0, 240.0)   # position angulaire des 3 bras (°)

    def direction(self, i: int):
        """Vecteur unitaire radial (ux, uy) du bras i."""
        phi = math.radians(self.arm_angles[i])
        return math.cos(phi), math.sin(phi)


def rotation(theta_x: float, theta_y: float, v):
    """Applique R = Ry(θx)·Rx(-θy) au vecteur v = (x, y, z)."""
    x, y, z = v
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    y1, z1 = cy * y + sy * z, -sy * y + cy * z          # Rx(-θy)
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    return cx * x + sx * z1, y1, -sx * x + cx * z1      # Ry(θx)


def plate_points(g: Geometry, theta_x: float, theta_y: float, h: float):
    """Positions des 3 rotules côté plateau."""
    points = []
    for i in range(3):
        ux, uy = g.direction(i)
        px, py, pz = rotation(theta_x, theta_y, (g.attachment_radius * ux, g.attachment_radius * uy, 0.0))
        points.append((px, py, pz + h))
    return points


def arm_tip(g: Geometry, i: int, alpha: float):
    """Position de la rotule côté bras, pour le bras i à l'angle α."""
    ux, uy = g.direction(i)
    r = g.base_radius + g.arm_length * math.cos(alpha)
    return r * ux, r * uy, g.arm_length * math.sin(alpha)


def inverse_kinematics(g: Geometry, theta_x: float, theta_y: float, h: float = None):
    """Angles (α1, α2, α3) des moteurs pour obtenir la pose voulue.

    Pour chaque bras : on note D = P - B (rotule plateau moins axe moteur), du et dz ses
    composantes radiale et verticale. La contrainte |P - A(α)| = l2 donne
        du·cos α + dz·sin α = K,   avec K = (|D|² + l1² - l2²) / (2·l1)
    soit ρ·cos(α - γ) = K, avec ρ = √(du² + dz²) et γ = atan2(dz, du) :
        α = γ - arccos(K / ρ)       (branche « bras vers l'extérieur »)
    Lève ValueError si la pose est hors de portée (|K| > ρ).
    """
    h = g.height if h is None else h
    l1, l2 = g.arm_length, g.rod_length
    alphas = []
    for i, (px, py, pz) in enumerate(plate_points(g, theta_x, theta_y, h)):
        ux, uy = g.direction(i)
        dx, dy, dz = px - g.base_radius * ux, py - g.base_radius * uy, pz
        du = dx * ux + dy * uy
        K = (dx * dx + dy * dy + dz * dz + l1 * l1 - l2 * l2) / (2 * l1)
        rho = math.hypot(du, dz)
        if abs(K) > rho:
            raise ValueError(f"Pose hors de portée du mécanisme (bras {i + 1})")
        alphas.append(math.atan2(dz, du) - math.acos(K / rho))
    return tuple(alphas)


def _residuals(g: Geometry, alphas, q):
    """|P_i - A_i|² - l2² pour les 3 bras (nuls quand la pose q est compatible avec les angles)."""
    theta_x, theta_y, h = q
    res = []
    for i, P in enumerate(plate_points(g, theta_x, theta_y, h)):
        A = arm_tip(g, i, alphas[i])
        res.append(sum((p - a) ** 2 for p, a in zip(P, A)) - g.rod_length ** 2)
    return res


def _solve_3x3(J, b):
    """Résout J·x = b par la règle de Cramer."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(J)
    solution = []
    for k in range(3):
        m = [row[:] for row in J]
        for r in range(3):
            m[r][k] = b[r]
        solution.append(det(m) / d)
    return solution


def forward_kinematics(g: Geometry, alphas, initial_pose=None, tol: float = 1e-12, iterations: int = 30):
    """Pose (θx, θy, h) du plateau pour des angles moteurs donnés (méthode de Newton).

    depart : pose de départ (la précédente, en simulation : convergence en 1 ou 2 itérations).
    """
    q = list(initial_pose) if initial_pose is not None else [0.0, 0.0, g.height]
    eps = 1e-7
    for _ in range(iterations):
        r = _residuals(g, alphas, q)
        J = [[0.0] * 3 for _ in range(3)]
        for k in range(3):
            q2 = q[:]
            q2[k] += eps
            r2 = _residuals(g, alphas, q2)
            for i in range(3):
                J[i][k] = (r2[i] - r[i]) / eps
        dq = _solve_3x3(J, [-v for v in r])
        q = [a + b for a, b in zip(q, dq)]
        if max(abs(v) for v in dq) < tol:
            return tuple(q)
    raise RuntimeError("La cinématique directe n'a pas convergé")
