import numpy as np
import scipy
# from my_files.src.params import Params


def channel_equalizer(input_vec: np.ndarray):
    equalizer_func = 1
    output_vec = input_vec * equalizer_func
    return output_vec


def normalize_vec(vec: np.ndarray, params):
    return params.normalization_factor * vec


def unnormalize_vec(vec: np.ndarray, params):
    return vec / params.normalization_factor


def pass_through_channel(input_vec: np.ndarray):
    channel_func = 1
    output_vec = input_vec * channel_func
    return output_vec


def fold_subvectors(vec: np.ndarray, N: int = 100) -> np.ndarray:
    """

    :param vec: long vector to be sliced into multiple sub vectors
    :param N: desired output sub-vector len
    :return: multi vector with length of N - each
    """
    # extend to multiplication of N
    N_zeros = int(np.ceil(len(vec) / N) * N - len(vec))
    vec = np.pad(vec,[0, N_zeros])
    # slice it
    vec = vec.reshape([-1, N])
    return vec


def estimate_T(X: np.ndarray) -> int:
    """
    estimate T (sample time) by the IFFT
    :param X: vector in xi domain
    :return: int T -
    """

    x = scipy.fft.iffft(X)

    from my_files.src import params as p
    return p.Tmax
