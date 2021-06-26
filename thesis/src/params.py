import numpy as np

path = '/thesis/output/'
# network params
M = 16  # Modulation order
# numBits = 3e5  # Number of bits to process
sps = 4  # Number of samples per symbol (oversampling factor)

# pulse_shaping params:
filter_len = 10  # filter length in symbols
roll_off = 0.25  # filter roll-off factor

weird_factor = 1  # TODO: what is this?
Ts = 1  # symbol period
Fs = 24  # sampling rate

# params for rest
length_of_msg = 1024  # N_t
BW = 2
m_qam = 16

alpha = 2.0  # TODO: whats that?
beta = 0.55  # TODO: whats that?

normalization_factor = 1e-3
Tmax = 15

length_of_time = int(length_of_msg / np.log2(m_qam))
length_of_xi = length_of_time  # not sure if it can be different than N_t

gamma = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)  # TODO: whats that?
