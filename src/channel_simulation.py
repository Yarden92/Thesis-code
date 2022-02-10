import numpy as np

from src.channel_blocks import ChannelBlocks
from src.split_step_fourier import SplitStepFourier


class ChannelSimulator:
    Ts = 3  # symbol period
    over_sampling = 8  # sampling rate
    roll_off = 0.25  # filter roll-off factor

    @property
    def length_of_msg(self):
        return int(self.num_symbols * self.sps)  # N_t

    @property
    def sps(self):
        return int(np.log2(self.m_qam))  # samples per symbol (4)

    def __init__(self,
                 m_qam: int = 16,
                 num_symbols: float = 64,
                 normalization_factor: float = 1e-3,
                 p_0: float = 0.00064,
                 t0: float=125e-12,  # 1/50GHz (2*10^-11)
                 # t0: float = 30,
                 channel_func: SplitStepFourier = None,
                 verbose=True):
        # ~~~~~~~~~~~~~~~~~~~~~~~ Network Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.m_qam = m_qam
        self.num_symbols = num_symbols
        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Other Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.normalization_factor = normalization_factor
        self.t0 = t0

        # ~~~~~~~~~~~~~~~~~~~~~~ Channel Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.P_0 = p_0
        self.channel_func = channel_func or SplitStepFourier()
        print(f'number of iterations in split step algo: {self.channel_func.N}')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.verbose = verbose
        self.cb = ChannelBlocks(verbose)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.x = [np.array([])] * 11  # outputs from 11 steps
        self.modem = None
        self.h_rrc = None  # filter
        self.N_rrc = None  # filter len

        # NFT Params
        self.N_xi = 0  # (=M)
        self.N_time = 0  # (=D)
        self.tvec = None
        self.xivec = None
        self.BW = 0

    def iterate_through_channel(self):
        for i in range(11):
            step = getattr(self, f'step{i}')
            step()
        return self.evaluate()

    def step0(self):
        self.x[0] = self.cb.generate_message(self.length_of_msg, self.sps)

    def step1(self):
        self.x[1], self.modem = self.cb.modulate(self.x[0], self.m_qam)

    def step2(self):
        self.x[2] = self.cb.over_sample(self.x[1], self.over_sampling)

    def step3(self):
        self.x[3], self.N_rrc, self.h_rrc = self.cb.pulse_shape(self.x[2], self.roll_off, self.over_sampling, self.Ts)

    def step4(self):
        self.x[4] = self.cb.pre_equalize(self.x[3], self.normalization_factor, self.P_0)

    def step5(self):
        self.N_xi, self.N_time, self.tvec, self.xivec, self.BW, self.dt = self.cb.gen_nft_params(self.x[4], self.t0)
        self.channel_func.set_dt(self.dt)

        self.x[5] = self.cb.inft(self.x[4], self.tvec, self.xivec)

    def step6(self):
        self.x[6] = self.cb.channel(self.x[5], self.channel_func, self.P_0)


    def step7(self):
        self.x[7] = self.cb.nft(self.x[6], self.tvec, self.xivec, self.BW, self.N_xi, self.N_rrc, self.x[4])

    def step8(self):
        self.x[8] = self.cb.equalizer(self.x[7], self.normalization_factor)

    def step9(self):
        self.x[9] = self.cb.match_filter(self.x[8], self.h_rrc, self.N_rrc, self.over_sampling, self.m_qam, self.sps)

    def step10(self):
        self.x[10] = self.cb.demodulate(self.x[9], self.modem, self.length_of_msg)

    def evaluate(self):
        return self.cb.calc_ber(self.x[0], self.x[10], self.length_of_msg, self.sps)
