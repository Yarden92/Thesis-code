import numpy as np
from ModulationPy import ModulationPy
from matplotlib import pyplot as plt

from myFNFTpy.FNFTpy import nsev_inverse_xi_wrapper, nsev_inverse, nsev
from lib.rrcos import rrcosfilter
from src.visualizer import Visualizer


class ChannelBlocks:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def generate_message(self, length_of_msg, sps):  # step0
        y = np.random.choice([0, 1], size=length_of_msg)  # generate random message - binary vec [0,1,0,...]
        if self.verbose:
            Visualizer.print_bits(y, sps, 'message before channel')

        return y

    def modulate(self, x, m_qam):  # step 1
        modem = ModulationPy.QAMModem(m_qam, soft_decision=False)
        y = modem.modulate(x)  # r[xi,0] = [1+j,-1+j,1-j,...]
        if self.verbose:
            Visualizer.plot_constellation_map_with_points(y, m_qam, 'clean before channel')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y[0:30]), function='stem', name='i (real)', ax=ax1, hold=1)
            Visualizer.my_plot(np.imag(y[0:30]), function='stem', name='q (imag)', ax=ax2)
            print(f'num of symbols = {len(y)}')

        return y, modem

    def over_sample(self, x, over_sampling):  # step 2
        y = np.zeros(over_sampling * (len(x) - 1) + 1, dtype=np.complex64)
        y[::over_sampling] = x
        if self.verbose:
            Visualizer.my_plot(np.real(y[0:50]), name='zero padded - i (real)', function='stem')
            print(f'vec length = {len(y)}, over_sampling period = {over_sampling}')

        return y

    def pulse_shape(self, x, roll_off, over_sampling, Ts):  # step 3
        # design the filter length to complete x to be power of 2
        lenx = len(x)
        desired_len = int(2 ** np.ceil(np.log2(lenx)))  # the next power of 2
        if desired_len - lenx < 100:
            desired_len = int(2 ** (1 + np.ceil(np.log2(lenx))))  # the next next power of 2

        L_rrc = int(np.ceil(desired_len - lenx)) + 1
        alpha: float = roll_off  # default = 0.25
        fs = over_sampling / Ts
        h_ind, h_rrc = rrcosfilter(L_rrc, alpha, Ts, fs)
        y = np.convolve(h_rrc, x)  # Waveform with PSF

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(h_rrc), name='h_rrc filter', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(h_rrc[L_rrc // 2:L_rrc // 2 + 24] ** 2), name='h_rrc^2 - zoomed in', ax=ax2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y[:]), name='X(xi) * h(xi)', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y[:L_rrc]), name='zoom in', ax=ax2)

            print(f'filter len = {L_rrc}, signal len = {len(y)}')
        return y, L_rrc, h_rrc

    def pre_equalize(self, x, normalization_factor):  # step 4
        # normalize vector:
        y = normalization_factor * x
        if self.verbose:
            # visualizer.my_plot(xi_axis, np.real(y),
            Visualizer.my_plot(np.real(y), legend=['Real{X}'], name='signal after pre equalizer', xlabel='xi')
            print(f'signal len = {len(y)}')

        return y

    def gen_nft_params(self, x, Tmax):  # step 5.1
        # some basic params for NFT
        N_xi = len(x)  # (=M)
        N_time = int(2 ** np.floor(np.log2(N_xi / 2)))  # (=D)
        tvec = np.linspace(-Tmax, Tmax, N_time)
        rv, xi = nsev_inverse_xi_wrapper(N_time, tvec[0], tvec[-1], N_xi)
        xivec = xi[0] + np.arange(N_xi) * (xi[1] - xi[0]) / (N_xi - 1)
        BW = xivec.max()

        return N_xi, N_time, tvec, xivec, BW

    def inft(self, x, tvec, xivec):  # step 5.2
        # INFT
        contspec = x
        bound_states = []  # np.array([0.7j, 1.7j])
        discspec = []  # [1.0, -1.0]

        cst = 1  # continuous spectrum type - default is None
        dst = 0  # default is None

        res = nsev_inverse(xivec, tvec, contspec, bound_states, discspec,
                           cst=cst, dst=dst)

        assert res['return_value'] == 0, "INFT failed"

        y = res['q']  # q[t,0]

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(tvec, np.abs(y), name=f'|x(t)|', xlabel='t', ax=ax1,
                               hold=1)
            Visualizer.my_plot(np.real(y), name=f'real(x(t))', ax=ax2)
            print(f'length of INFT(x) = {len(y)}')

        return y

    def channel(self, x, channel_func):  # step 6
        y = channel_func(x)

        if self.verbose:
            Visualizer.my_plot(np.abs(y),name=f'|x(t)|')
        return y

    def nft(self, x, tvec, xivec, BW, N_xi, L_rrc, ref_y=None):  # step 7
        res = nsev(x, tvec, Xi1=-BW, Xi2=BW, M=N_xi)
        assert res['return_value'] == 0, "NFT failed"
        y = res['cont_ref']  # r[xi,L]
        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(xivec, np.real(y),
                               name=f'real{{X(xi)}}, (BW={BW:.0f})',
                               ylabel='real{X(xi)}', xlabel='xi', ax=ax1, hold=1)

            Visualizer.my_plot(np.real(y)[:L_rrc], name='zoom in', ax=ax2)
            if ref_y is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
                Visualizer.my_plot(xivec, np.real(ref_y), name='reference before INFT',
                                   xlabel='xi', ax=ax1, hold=1)
                Visualizer.my_plot(np.real(ref_y)[:L_rrc], name='reference zoom in', ax=ax2)
        return y

    def equalizer(self, x, normalization_factor):  # step 8
        # unnormalize
        y1 = x / normalization_factor

        # channel equalizer (nothing)
        equalizer_func = 1
        y2 = y1 * equalizer_func
        return y2

    def match_filter(self, x, h_rrc, L_rrc, over_sampling, m_qam=None, sps=None):  # step 9
        y1 = np.convolve(x, h_rrc)

        # sampling the analog vector into discrete bits
        start = L_rrc
        stop = - L_rrc + over_sampling
        step = over_sampling
        y2 = y1[start:stop:step] / over_sampling

        if self.verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y1), name='analog signal after another conv', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y1[0:L_rrc * 2]), name='zoom in', ax=ax2)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y2), name='sampled bits', function='stem', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y2)[0:50], name='zoom in', function='stem', ax=ax2)
            Visualizer.plot_constellation_map_with_points(y2, m_qam, 'after depulse shaping')
            Visualizer.eye_diagram(y2, sps=sps)
        return y2

    def demodulate(self, x, modem, length_of_msg):  # step 10
        y = modem.demodulate(x)
        assert len(y) == length_of_msg, \
            f"oh no, the outcome is not {length_of_msg}, but {len(y)}"
        return y

    def calc_ber(self, x_in: np.ndarray, x_out: np.ndarray, length_of_msg, sps):
        num_errors = (x_in != x_out).sum()
        ber = num_errors / length_of_msg

        if self.verbose:
            Visualizer.print_bits(x_out, sps, 'message after channel')
            print(f'ber = {ber} = {num_errors}/{length_of_msg}')
        return ber, num_errors
