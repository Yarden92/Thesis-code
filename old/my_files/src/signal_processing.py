import numpy as np
from scipy.signal import butter, lfilter

from old.my_files.src import NFT


def channel_equalizer(input_vec: np.ndarray):
    equalizer_func = 1
    output_vec = input_vec * equalizer_func
    return output_vec


def normalize_vec(vec: np.ndarray, factor):
    return factor * vec


def unnormalize_vec(vec: np.ndarray, factor):
    return vec / factor


def pass_through_channel(input_vec: np.ndarray, channel_func):
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
    vec = np.pad(vec, [0, N_zeros])
    # slice it
    vec = vec.reshape([-1, N])
    return vec


def estimate_T(X: np.ndarray) -> int:
    """
    estimate T (sample time) by the IFFT
    :param X: vector in xi domain
    :return: int T - the time period
    """

    x = np.fft.ifft(X)

    # TODO: find the part where the signal power goes zero..
    return 200


def pad_vec(x: np.ndarray, before: float, after: float, padding_value: float = 0) -> np.ndarray:
    """
    padding a vector before and after with constant value
    :param x: target vector to be padded
    :param before: number of values to pad before the vector
    :param after: number of values to pad after the vector
    :param padding_value: the value that will be padded
    :return: padded vector
    """
    # linear_ramp smoothens the padding, if you prefer straight cut use 'constant'
    x = np.pad(x, (before, after), 'linear_ramp', end_values=(padding_value, padding_value))

    return x


def bpf(X_xi: np.ndarray, xi_axis: np.ndarray, fstart: float, fstop: float,
        order: int = 5) -> np.ndarray:
    """
    band pass filter a signal
    :param X_xi: signal in xi domain (1D vec)
    :param xi_axis: vector of the x_axis values
    :param fstart: starting freq of the BPF
    :param fstop: stopping freq of the BPF
    :return: filtered signal
    """
    N = len(X_xi)
    assert len(xi_axis) == N, "given xi_axis and the signal in xi domain must be same length"

    index_start = find_nearest(xi_axis, fstart)
    index_stop = find_nearest(xi_axis, fstop)

    Wn_start = index_start/N
    Wn_stop = index_stop/N

    # nyq = 0.5 * fs
    # low = lowcut / nyq
    # high = highcut / nyq
    b, a = butter(order, (Wn_start, Wn_stop), btype='bandpass')

    X_t = NFT.IFFT(X_xi)
    y = lfilter(b, a, X_t)
    y_xi = NFT.FFT(y)

    return y_xi


def find_nearest(array: np.ndarray, target_value: float) -> int:
    """
    searches in `array` for the closest value of target_value
    :param array: some numpy vector of values
    :param target_value: the value we want to look for at array
    :return: the index of the nearest value
    """
    idx = (np.abs(array - target_value)).argmin()
    return idx


def up_sample(x:np.ndarray, factor: int) -> np.ndarray:
    """
    up sample a vector by factor, using zero padding in between each sample
    :param x: signal vector
    :param factor: number of times the vector should by up sampled by
    :return: up sampled vector
    """
    y = np.zeros(factor * (len(x) - 1) + 1, dtype=np.complex64)
    y[::factor] = x
    return y

def fix_length(x:np.ndarray) -> np.ndarray:
    """
    pad with zeros to make length of x to be power of 2
    :param x: some vector
    :return: longer vector of x, with zeros at the end
    """
    current_length = len(x)
    desired_length = int(2**np.ceil(np.log2(current_length)))
    num_of_zeros = desired_length - current_length
    num_of_zeros_after = num_of_zeros//2
    num_of_zeros_before = num_of_zeros - num_of_zeros_after
    x = np.pad(x,(num_of_zeros_before,num_of_zeros_after),'constant',constant_values=0)
    return x