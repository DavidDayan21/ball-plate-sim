"""Gains of the PID of one axis by LQR (linear-quadratic regulator).

Model of one axis, linearized: state s = (z, x, v) with z = ∫x·dt, ẋ = v and v̇ = c·g·θ.
The LQR feedback θ = -K·s minimizes ∫(sᵀ·Q·s + R·θ²)·dt. With the setpoint at 0 (e = -x), it is
exactly a PID: K = (Ki, Kp, Kd).

Weights by Bryson's rule: a position error of position_scale, an integral of the error of
position_scale·integral_time and a tilt of tilt_scale cost as much as one another. The smaller
position_scale, the stiffer the loop; integral_time sets how fast the integral acts.
"""
import numpy as np


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Gain K of the continuous LQR (u = -K·s).

    The solution P of the Riccati equation AᵀP + PA - PBR⁻¹BᵀP + Q = 0 is read from the stable
    invariant subspace of the Hamiltonian matrix [[A, -BR⁻¹Bᵀ], [-Q, -Aᵀ]]: if its n stable
    eigenvectors are stacked as [X1; X2], then P = X2·X1⁻¹. K = R⁻¹BᵀP.
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    H = np.block([[A, -B @ R_inv @ B.T], [-Q, -A.T]])
    values, vectors = np.linalg.eig(H)
    stable = vectors[:, values.real < 0]
    P = np.real(stable[n:] @ np.linalg.inv(stable[:n]))
    return R_inv @ B.T @ P


def pid_gains(model_gain: float, position_scale: float, integral_time: float, tilt_scale: float) -> tuple:
    """(Kp, Ki, Kd) of one axis. model_gain = c·g, in (m/s²)/rad."""
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    B = np.array([[0.0], [0.0], [model_gain]])
    Q = np.diag([1 / (position_scale * integral_time) ** 2, 1 / position_scale ** 2, 0.0])
    ki, kp, kd = lqr(A, B, Q, np.array([[1 / tilt_scale ** 2]]))[0]
    return float(kp), float(ki), float(kd)
