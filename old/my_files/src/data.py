import numpy as np
from ModulationPy import ModulationPy


# from my_files.src.params

def generate_random_msg(N: int) -> np.ndarray:
    """

    :param N: length of msg vector
    :return:
    """
    return np.random.choice([0, 1], size=N)


def data_modulation(binary_vec: np.ndarray, M_QAM: int) -> np.ndarray:
    """
    binary bits -> [Modulator] -> i_q_samples_vector
    The modulator takes chunks of M binary bits and map them to the i_q plane (complex values, amp+phase)
    :param binary_vec: a stream of binary bits, where each sub group of M bits is considered symbol
    :param M_QAM: the order of the QAM - num of bits per symbol
    :return modulated_vec: vec of complex i_q values, each represents M_QAM bits, with len divided by M_QAM
    """
    modem = ModulationPy.QAMModem(M_QAM,gray_map=True,)
    modulated_vec = modem.modulate(binary_vec)
    return modulated_vec


def data_demodulation(modulated_vec: np.ndarray, M_QAM: int) -> np.ndarray:
    """
    i_q_samples_vec -> [Demodulator] -> binary_bits
    The demodulator takes vector of complex values, and convert each one to M binary bits
    :param modulated_vec: vector of complex i_q values
    :param M_QAM: the order of the QAM - num of bits per symbol
    :return binary_vec: multiple subgroups of M binary bits - concatenated into long vector.
    """
    modem = ModulationPy.QAMModem(M_QAM, gray_map=True, bin_output=True)
    binary_vec = modem.demodulate(modulated_vec)
    return binary_vec
