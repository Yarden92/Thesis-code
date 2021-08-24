import numpy as np

visualization_path = '/thesis/output/'
# network params
M = 16  # Modulation order
bps = int(np.log2(M))  # 4
# numBits = 3e5  # Number of bits to process
num_symbols = 2000
# sps = 4  # Number of samples per symbol (oversampling factor)

# pulse_shaping params:
filter_len = 49  # filter length in symbols
roll_off = 0.25  # filter roll-off factor
# NFT_size = 100

# weird_factor = 1  # TODO: what is this?
Ts = 1  # symbol period
Fs = 4  # sampling rate

# params for rest
length_of_msg = int(num_symbols * bps)  # N_t
# BW = 2
m_qam = M

# alpha = 2.0  # TODO: whats that?
# beta = 0.55  # TODO: whats that?

normalization_factor = 1e-3
# normalization_factor = 1
Tmax = 15

# length_of_time = int(length_of_msg / bps)
# length_of_xi = length_of_time  # not sure if it can be different than N_t

# gamma = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)  # TODO: whats that?

channel_func = 1



# SETTINGS
plot_vec_after_creation = True
