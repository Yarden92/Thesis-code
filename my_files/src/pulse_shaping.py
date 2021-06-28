import numpy as np
from commpy.filters import rrcosfilter

from my_files.src import visualizer, params as p


def pulse_shaping(i_q_samples: np.ndarray) -> np.ndarray:

    N = p.filter_len
    k = np.log2(p.M)  # Number of bits per symbol

    time_indices, h_rrc = rrcosfilter(N, p.roll_off, p.Ts, p.Fs)
    visualizer.my_plot(time_indices,h_rrc,name='rrc_filter',xlabel='time',ylabel='h(t)')

    qW = np.convolve(h_rrc, i_q_samples)  # Waveform with PSF


    return qW
