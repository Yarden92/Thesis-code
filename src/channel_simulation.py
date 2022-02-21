import numpy as np
from matplotlib import pyplot as plt

from src.channel_blocks import ChannelBlocks
from src.split_step_fourier import SplitStepFourier
from src.visualizer import Visualizer


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
                 dt: float = 1e-12,
                 channel_func: SplitStepFourier = None,
                 verbose=True,
                 test_verbose=False
                 ):
        # ~~~~~~~~~~~~~~~~~~~~~~~ Network Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.m_qam = m_qam
        self.num_symbols = num_symbols
        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Other Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.normalization_factor = normalization_factor
        # self.t0 = t0
        self.dt = dt

        # ~~~~~~~~~~~~~~~~~~~~~~ Channel Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.P_0 = p_0
        self.channel_func = channel_func or SplitStepFourier()
        print(f'number of iterations in split step algo: {self.channel_func.N}')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.verbose = verbose
        self.test_verbose = test_verbose
        self.cb = ChannelBlocks()

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
        self.step0_gen_msg()
        self.step1_modulate()
        self.step2_over_sample()
        self.step3_pulse_shaping()
        self.step4_pre_equalize()
        self.step5_inft()
        self.step6_channel()
        self.step7_nft()
        self.step8_equalize()
        self.step9_match_filter()
        self.step10_demodulate()

        return self.evaluate()

    def step0_gen_msg(self):
        self.x[0] = self.cb.generate_message(self.length_of_msg)
        if self.verbose:
            Visualizer.print_bits(self.x[0], self.sps, 'message before channel')

    def step1_modulate(self):
        self.x[1], self.modem = self.cb.modulate(self.x[0], self.m_qam)
        if self.verbose:
            Visualizer.plot_constellation_map_with_points(self.x[1], self.m_qam, 'clean before channel')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(self.x[1][0:30]), function='stem', name='i (real)', ax=ax1, hold=1)
            Visualizer.my_plot(np.imag(self.x[1][0:30]), function='stem', name='q (imag)', ax=ax2)
            print(f'num of symbols = {len(self.x[1])}')

        self._test_step1()

    def _test_step1(self):
        x0_reconst = self.modem.demodulate(self.x[1])
        num_errors = (x0_reconst != self.x[0]).sum()

        assert num_errors == 0, \
            f"test modulate failed, there are {num_errors} errors instead of 0"

        if self.test_verbose:
            Visualizer.print_bits(x0_reconst, self.sps, 'test reconstructed msg')
            Visualizer.print_bits(self.x[0], self.sps, 'message before channel')
            print(f'ber = {num_errors / self.length_of_msg} = {num_errors}/{self.length_of_msg}')

    def step2_over_sample(self):
        self.x[2] = self.cb.over_sample(self.x[1], self.over_sampling)
        if self.verbose:
            Visualizer.my_plot(np.real(self.x[2][0:50]), name='zero padded - i (real)', function='stem')
            print(f'vec length = {len(self.x[2])}, over_sampling period = {self.over_sampling}')

    def step3_pulse_shaping(self):
        self.x[3], self.N_rrc, self.h_rrc = self.cb.pulse_shape(self.x[2], self.roll_off, self.over_sampling, self.Ts)

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(self.h_rrc), name='h_rrc filter', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(self.h_rrc[self.N_rrc // 2:self.N_rrc // 2 + 24] ** 2),
                               name='h_rrc^2 - zoomed in', ax=ax2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(self.x[3][:]), name='X(xi) * h(xi)', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(self.x[3][:self.N_rrc]), name='zoom in', ax=ax2)

            print(f'filter len = {self.N_rrc}, signal len = {len(self.x[3])}')

        self._test_step3()

    def _test_step3(self):

        xx41 = np.convolve(self.x[3], self.h_rrc)

        # sampling the analog vector into discrete bits
        N_rrc = len(self.h_rrc)
        start = N_rrc
        stop = - N_rrc
        step = self.over_sampling
        xx42 = xx41[start:stop:step] / self.over_sampling
        x0_reconst = self.modem.demodulate(xx42)
        num_errors = (x0_reconst != self.x[0]).sum()

        assert num_errors == 0, \
            f"test pulse shaping failed, there are {num_errors} errors instead of 0"

        if self.test_verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(xx41), name='analog signal after another conv', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(xx41[:N_rrc * 2]), name='zoom in', ax=ax2)

            Visualizer.my_plot(np.real(xx42[0:50]), name='sampled bits', function='stem')

            Visualizer.plot_constellation_map_with_points(xx42, self.m_qam, 'after depulse shaping')
            Visualizer.eye_diagram(xx42, sps=self.sps)

    def step4_pre_equalize(self):
        self.N_xi, self.N_time, self.tvec, self.xivec, self.BW = self.cb.gen_nft_params(len(self.x[3]), self.dt)

        if self.verbose:
            print(
                f'xi ∈ [{self.xivec[0] / 1e9:.2f}:{self.xivec[-1] / 1e9:.2f}] GHz ,\t N_xi   (=M) = {self.N_xi}\n'
                f't  ∈ [{self.tvec[0] * 1e12:.2f}:{self.tvec[-1] * 1e12:.2f}] ps    ,\t N_time (=D) = {self.N_time}\n'
                f'BW = {self.BW / 1e9:.2f} GHz'
            )

        self.x[4] = self.cb.pre_equalize(self.x[3], self.normalization_factor)

        if self.verbose:
            # TODO: print signal in xi domain with xi_vec axis, just like the prints after inft
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4))
            Visualizer.my_plot(self.xivec, np.abs(self.x[4]), name=f'|X(xi)|',xlabel='xi [Hz]',ax=ax1, hold=1)
            Visualizer.my_plot(np.real(self.x[4]), name='Real{X}', xlabel='index',ax=ax2)
            print(f'signal len = {len(self.x[4])}')

    def step5_inft(self):
        self.x[5] = self.cb.inft(self.x[4], self.tvec, self.xivec, self.P_0)

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(self.tvec * 1e0, np.abs(self.x[5]), name=f'|x(t)|', xlabel='t [s]', ax=ax1,
                               hold=1)
            Visualizer.my_plot(np.real(self.x[5]), name=f'real(x(t))', ax=ax2)
            print(f'length of INFT(x) = {len(self.x[5])}')


            Visualizer.print_signal_specs(self.x[5], self.tvec)

    def step6_channel(self):
        self.x[6] = self.cb.channel(self.x[5], self.channel_func, self.P_0)

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(self.tvec, np.abs(self.x[6]), name=f'|x(t)|', xlabel='t[s]', ax=ax1,
                               hold=1)
            Visualizer.my_plot(np.real(self.x[6]), name=f'real(x(t))',xlabel='index', ax=ax2)

    def step7_nft(self):
        self.x[7] = self.cb.nft(self.x[6], self.tvec, self.BW, self.N_xi)

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(self.xivec, np.real(self.x[7]),
                               name=f'real{{X(xi)}}, (BW={self.BW:.0f})',
                               ylabel='real{X(xi)}', xlabel='xi', ax=ax1, hold=1)

            Visualizer.my_plot(np.real(self.x[7])[:self.N_rrc], name='zoom in', ax=ax2)
            if self.x[4] is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                Visualizer.my_plot(self.xivec, np.real(self.x[4]), name='reference before INFT',
                                   xlabel='xi', ax=ax1, hold=1)
                Visualizer.my_plot(np.real(self.x[4])[:self.N_rrc], name='reference zoom in', ax=ax2)

    def step8_equalize(self):
        self.x[8] = self.cb.equalizer(self.x[7], self.normalization_factor)

    def step9_match_filter(self):
        self.x[9], y1 = self.cb.match_filter(self.x[8], self.h_rrc, self.N_rrc, self.over_sampling)
        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y1), name='analog signal after another conv', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y1[0:self.N_rrc * 2]), name='zoom in', ax=ax2)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(self.x[9]), name='sampled bits', function='stem', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(self.x[9])[0:50], name='zoom in', function='stem', ax=ax2)
            Visualizer.plot_constellation_map_with_points(self.x[9], self.m_qam, 'after depulse shaping')
            Visualizer.eye_diagram(self.x[9], sps=self.sps)
            print(f'num of sampled symbols = {len(self.x[9])} ')

    def step10_demodulate(self):
        self.x[10] = self.cb.demodulate(self.x[9], self.modem, self.length_of_msg)
        if self.verbose:
            Visualizer.print_bits(self.x[10], self.sps, 'message after channel')

    def evaluate(self):
        ber, num_errors = self.cb.calc_ber(self.x[0], self.x[10], self.length_of_msg, self.sps)
        if self.verbose:
            print(f'ber = {ber} = {num_errors}/{self.length_of_msg}')

        return ber, num_errors
