import numpy as np


class SP:
    @staticmethod
    def signal_power(x: np.ndarray) -> float:
        return np.linalg.norm(x,ord=2)

    @staticmethod
    def peak(x: np.ndarray) -> float:
        return np.max(np.abs(x))
