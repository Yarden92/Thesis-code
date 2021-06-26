import numpy as np

# from thesis.src.params import Params


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
