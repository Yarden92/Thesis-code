import numpy as np
from commpy.filters import rrcosfilter

from thesis.src import params as p


def pulse_shaping(i_q_samples: np.ndarray) -> np.ndarray:

    N = len(i_q_samples)
    k = np.log2(p.M)  # Number of bits per symbol

    time_indices, h_rrc = rrcosfilter(N, p.roll_off, p.Ts, p.Fs)
    qW = np.convolve(h_rrc, i_q_samples)  # Waveform with PSF

    return qW
