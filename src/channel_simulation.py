import matplotlib.pyplot as plt
import numpy as np
from ModulationPy import ModulationPy

from FNFTpy import nsev_inverse_xi_wrapper, nsev_inverse, nsev
from src.visualizer import Visualizer
from lib.rrcos import rrcosfilter


class Step0:
    @staticmethod
    def generate_message(p):
        x = np.random.choice([0, 1], size=p.length_of_msg)  # generate random message - binary vec [0,1,0,...]
        if p.plot_vec_after_creation:
            Visualizer.print_bits(x, p.sps, 'message before channel')
        return x


class Step1:
    @staticmethod
    def modulate(x, p):
        modem = ModulationPy.QAMModem(p.m_qam, soft_decision=False)
        y = modem.modulate(x)  # r[xi,0] = [1+j,-1+j,1-j,...]
        if p.plot_vec_after_creation:
            Visualizer.plot_constellation_map_with_points(y, p.m_qam, 'clean before channel')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y[0:30]), function='stem', name='i (real)', ax=ax1, hold=1)
            Visualizer.my_plot(np.imag(y[0:30]), function='stem', name='q (imag)', ax=ax2)
            print(f'num of symbols = {len(y)}')
        return y, modem


class Step2:
    @staticmethod
    def over_sample(x, p):
        y = np.zeros(p.over_sampling * (len(x) - 1) + 1, dtype=np.complex64)
        y[::p.over_sampling] = x
        if p.plot_vec_after_creation:
            Visualizer.my_plot(np.real(y[0:50]), name='zero padded - i (real)', function='stem')
            print(f'vec length = {len(y)}, over_sampling period = {p.over_sampling}')
        return y


class Step3:
    @staticmethod
    def pulse_shape(x, p):
        # design the filter length to complete x to be power of 2
        lenx = len(x)
        desired_len = int(2 ** np.ceil(np.log2(lenx)))  # the next power of 2
        if desired_len - lenx < 100:
            desired_len = int(2 ** (1 + np.ceil(np.log2(lenx))))  # the next next power of 2

        L_rrc = int(np.ceil(desired_len - lenx)) + 1
        alpha: float = p.roll_off  # default = 0.25
        fs = p.over_sampling / p.Ts
        h_ind, h_rrc = rrcosfilter(L_rrc, alpha, p.Ts, fs)
        y = np.convolve(h_rrc, x)  # Waveform with PSF

        if p.plot_vec_after_creation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(h_rrc), name='h_rrc filter', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(h_rrc[L_rrc // 2:L_rrc // 2 + 24] ** 2),
                               name='h_rrc^2 - zoomed in', ax=ax2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y[:]), name='X(xi) * h(xi)', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y[:L_rrc]), name='zoom in', ax=ax2)

            print(f'filter len = {L_rrc}, signal len = {len(y)}')
        return y, h_rrc, L_rrc


class Step4:
    @staticmethod
    def pre_equalize(x, p):
        # normalize vector:
        y = p.normalization_factor * x
        if p.plot_vec_after_creation:
            # visualizer.my_plot(xi_axis, np.real(y),
            Visualizer.my_plot(np.real(y), legend=['Real{X}'], name='signal after pre equalizer', xlabel='xi')
            print(f'signal len = {len(y)}')
        return y


class Step5:
    @staticmethod
    def add_nft_params(x, p, o):
        # some basic params for NFT
        o.N_xi = len(x)  # (=M)
        o.N_time = int(2 ** np.floor(np.log2(o.N_xi / 2)))  # (=D)
        o.tvec = np.linspace(-p.Tmax, p.Tmax, o.N_time)
        rv, xi = nsev_inverse_xi_wrapper(o.N_time, o.tvec[0],
                                         o.tvec[-1], o.N_xi)
        o.xivec = xi[0] + np.arange(o.N_xi) * (xi[1] - xi[0]) / (o.N_xi - 1)
        o.BW = o.xivec.max()

    @staticmethod
    def inft(x, p, o):
        # INFT
        contspec = x
        bound_states = []  # np.array([0.7j, 1.7j])
        discspec = []  # [1.0, -1.0]

        cst = 1  # continuous spectrum type - default is None
        dst = 0  # default is None

        res = nsev_inverse(o.xivec, o.tvec, contspec, bound_states, discspec,
                           cst=cst, dst=dst)

        assert res['return_value'] == 0, "INFT failed"

        y = res['q']  # q[t,0]

        if p.plot_vec_after_creation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(o.tvec, np.abs(y), name=f'|x(t)|, (T={p.Tmax})', xlabel='t', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y), name=f'real(x(t))', ax=ax2)
            print(f'length of INFT(x) = {len(y)}')
        return y


class Step6:
    @staticmethod
    def channel(x, p):
        channel_func = 1
        y = x * channel_func  # q[t,L]
        return y


class Step7:
    @staticmethod
    def nft(x, p, o):
        res = nsev(x, o.tvec, Xi1=-o.BW, Xi2=o.BW, M=o.N_xi)
        assert res['return_value'] == 0, "NFT failed"
        y = res['cont_ref']  # r[xi,L]
        if p.plot_vec_after_creation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(o.xivec, np.real(y),
                               name=f'real{{X(xi)}}, (BW={o.BW:.0f})',
                               ylabel='real{X(xi)}', xlabel='xi', ax=ax1, hold=1)

            Visualizer.my_plot(np.real(y)[:o.L_rrc], name='zoom in', ax=ax2)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(o.xivec, np.real(o.x[4]), name='reference before INFT',
                               xlabel='xi', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(o.x[4])[:o.L_rrc], name='reference zoom in', ax=ax2)
        return y


class Step8:
    @staticmethod
    def equalizer(x, p):
        # unnormalize
        y1 = x / p.normalization_factor

        # channel equalizer (nothing)
        equalizer_func = 1
        y2 = y1 * equalizer_func
        return y2


class Step9:
    @staticmethod
    def match_filter(x, p, o):
        y1 = np.convolve(x, o.h_rrc)

        # sampling the analog vector into discrete bits
        start = o.L_rrc
        stop = - o.L_rrc + p.over_sampling
        step = p.over_sampling
        y2 = y1[start:stop:step] / p.over_sampling

        if p.plot_vec_after_creation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y1), name='analog signal after another conv', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y1[0:o.L_rrc * 2]), name='zoom in', ax=ax2)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            Visualizer.my_plot(np.real(y2), name='sampled bits', function='stem', ax=ax1, hold=1)
            Visualizer.my_plot(np.real(y2)[0:50], name='zoom in', function='stem', ax=ax2)
            Visualizer.plot_constellation_map_with_points(y2, p.m_qam, 'after depulse shaping')
            Visualizer.eye_diagram(y2, sps=p.sps)
        return y2


class Step10:
    @staticmethod
    def demodulate(x, p, o, org_msg, modem):
        y = o.modem.demodulate(x)
        assert len(y) == p.length_of_msg, \
            f"oh no, the outcome is not {p.length_of_msg}, but {len(y)}"
        num_errors = (y != org_msg).sum()
        ber = num_errors / p.length_of_msg

        if p.plot_vec_after_creation:
            Visualizer.print_bits(y, p.sps, 'message after channel')
            print(f'ber = {ber} = {num_errors}/{p.length_of_msg}')
        return y, ber, num_errors
