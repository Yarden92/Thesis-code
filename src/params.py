import numpy as np


class Params:
    # ~~~~~~~~~~~~~~~~~~~~~ Pulse Shaping Params ~~~~~~~~~~~~~~~~~~~~~~
    Ts = 3  # symbol period
    over_sampling = 8  # sampling rate
    roll_off = 0.25  # filter roll-off factor

    # BW = 700*2*np.pi # <- its max(xivec)

    def __init__(self,
                 m_qam=16,
                 num_symbols=64,
                 normalization_factor=1e-3,
                 Tmax=30,
                 plot_vec_after_creation=True):
        # ~~~~~~~~~~~~~~~~~~~~~~~ Network Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.m_qam = m_qam
        self.num_symbols = num_symbols
        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Other Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.normalization_factor = normalization_factor
        self.Tmax = Tmax

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.plot_vec_after_creation = plot_vec_after_creation

    @property
    def length_of_msg(self):
        return int(self.num_symbols * self.sps)  # N_t

    @property
    def sps(self):
        return int(np.log2(self.m_qam))  # samples per symbol (4)
