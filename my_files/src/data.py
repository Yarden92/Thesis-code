import numpy as np
from ModulationPy import ModulationPy


# from my_files.src.params


def generate_random_msg(N: int) -> np.ndarray:
    # N = length of msg vector
    return np.random.choice([0, 1], size=N)


def data_encoder(binary_msg: np.ndarray, params) -> np.array:
    modem = ModulationPy.QAMModem(params.m_qam)
    modulated_msg = modem.modulate(binary_msg)
    return modulated_msg
