import numpy as np

from my_files.src import visualizer


# from lib.packages.CommPy.build.lib.commpy.filters import rrcosfilter


def pulse_shaping(i_q_samples: np.ndarray, Ts: float, OSF: float,
                  filter_len: int = 49, alpha: float = 0.25) -> np.ndarray:
    """
    :param i_q_samples: the signal vector
    :param Ts: symbol period in time unit
    :param OSF: over sampling factor
    :param filter_len: filter length (recommended as the number of symbols)
    :param alpha: roll off factor [0:1]
    :return: filtered smooth vector after pulse shaping
    """

    time_indices, h_rrc = rrcosfilter(filter_len, alpha, Ts, OSF)
    visualizer.my_plot(time_indices, h_rrc, name='rrc_filter', xlabel='time', ylabel='h(t)')

    qW = np.convolve(h_rrc, i_q_samples)  # Waveform with PSF

    return qW


def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1 / float(Fs)
    time_idx = ((np.arange(N) - N / 2)) * T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * \
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + \
                        4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                       (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

    return time_idx, h_rrc


