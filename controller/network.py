"""Neural network of the AI brain: a function observation -> action, computed with NumPy alone.

The network is learned with PyTorch (training/), then saved as its weights: using it needs neither PyTorch nor
the training libraries. It is a stack of layers, y = W·x + b, each followed by tanh except the last one.
"""
import numpy as np


class Network:
    """layers: list of (W, b), W of shape (outputs, inputs). Called with one observation, returns the action."""

    def __init__(self, layers):
        self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in layers]

    def __call__(self, observation) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        for w, b in self.layers[:-1]:
            x = np.tanh(w @ x + b)
        w, b = self.layers[-1]
        return w @ x + b

    def save(self, path) -> None:
        """Writes the weights in a .npz file."""
        arrays = {f"{name}{k}": array for k, layer in enumerate(self.layers) for name, array in zip("wb", layer)}
        with open(path, "wb") as f:                     # an open file: NumPy would add ".npz" to another name
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path) -> "Network":
        with np.load(path) as arrays:
            return cls([(arrays[f"w{k}"], arrays[f"b{k}"]) for k in range(len(arrays.files) // 2)])
