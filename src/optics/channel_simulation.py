import json

import numpy as np
from matplotlib import pyplot as plt

from src.deep.standalone_methods import DataType
from src.general_methods.visualizer import Visualizer
from src.optics.channel_blocks import ChannelBlocks
from src.optics.split_step_fourier import SplitStepFourier


class ChannelSimulator:
    Ts = 3  # symbol period
    roll_off = 0.25  # filter roll-off factor

    def __init__(self,
                 m_qam: int = 16,
                 num_symbols: float = 64,
                 normalization_factor: float = 1e-3,
                 dt: float = 1e-12,
                 ssf: SplitStepFourier = None,
                 verbose=True,
                 test_verbose=False,
                 over_sampling=8  # sampling rate
                 ):
        # ~~~~~~~~~~~~~~~~~~~~~~~ Network Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert np.log2(m_qam)%2 == 0, f'm_qam must be even power of 2, got {m_qam} which is 2^{np.log2(m_qam)}'
        self.m_qam = m_qam
        self.num_symbols = num_symbols
        self.over_sampling = over_sampling
        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Other Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.normalization_factor = normalization_factor
        # self.t0 = t0
        self.dt = dt

        # ~~~~~~~~~~~~~~~~~~~~~~ Channel Params ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.P_0 = p_0
        self.ssf = ssf or SplitStepFourier()
        if verbose:
            print(f'number of iterations in split step algo: {self.ssf.N}')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.verbose = verbose
        self.test_verbose = test_verbose
        self.cb = ChannelBlocks()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.x = [np.array([])]*11  # outputs from 11 steps
        self._modem = None
        self._h_rrc = None  # filter
        self._N_rrc = None  # filter len

        # NFT Params
        self.N_xi = 0  # (=M)
        self.N_time = 0  # (=D)
        self.tvec = None
        self.xivec = None
        self.BW = 0

    @property
    def length_of_msg(self):
        return self.num_symbols*self.sps  # N_t

    @property
    def sps(self):
        return int(np.log2(self.m_qam))  # samples per symbol (4)

    @property
    def modem(self):
        if self._modem is None:
            # validate m_qam is even power of 2 (4, 16, 64, 256, ...)
            self._modem = self.cb.generate_modem(self.m_qam)
        return self._modem

    @property
    def N_rrc(self):
        if self._N_rrc is None:
            self._N_rrc, self._h_rrc = self.cb.gen_h_rrc(self.num_symbols*self.over_sampling, self.roll_off,
                                                         self.over_sampling, self.Ts)
        return self._N_rrc

    @property
    def h_rrc(self):
        if self._h_rrc is None:
            self._N_rrc, self._h_rrc = self.cb.gen_h_rrc(self.num_symbols*self.over_sampling, self.roll_off,
                                                         self.over_sampling, self.Ts)
        return self._h_rrc

    # _______________________________________________________________________

    def params_to_dict(self):
        return {
            'm_qam': self.m_qam,
            'num_symbols': self.num_symbols,
            'normalization_factor': self.normalization_factor,
            'dt': self.dt,
            'ssf': self.ssf.params_to_dict(),
        }

    @classmethod
    def from_dict(cls, _dict: dict, verbose=False):
        return cls(
            m_qam=_dict['m_qam'],
            num_symbols=_dict['num_symbols'],
            normalization_factor=_dict['normalization_factor'],
            dt=_dict['dt'],
            ssf=SplitStepFourier.from_dict(_dict['ssf'], verbose),
            verbose=verbose,
            test_verbose=verbose
        )

    def __str__(self):
        return json.dumps(self.params_to_dict(), indent=4)

    def iterate_through_channel(self):
        # returns [ber, num_errors]
        self.step0_gen_msg()
        self.step1_modulate()
        self.step2_over_sample()
        self.step3_pulse_shaping()
        self.step4_pre_equalize() # -> x[4] (clean spectrum)
        self.step5_inft()
        self.step6_channel()
        self.step7_nft()        # -> x[7] (dirty spectrum)
        self.step8_equalize() 
        self.step9_match_filter()
        self.step10_demodulate()

        return self.evaluate()

    def steps8_to_10(self, x):
        self.x[7] = x
        self.step8_equalize()
        self.step9_match_filter()
        self.step10_demodulate()
        return self.x[10]

    def steps10(self, x):
        self.x[9] = x
        self.step10_demodulate()
        return self.x[10]

    def gen_io_data(self, type=DataType.spectrum) -> (np.ndarray, np.ndarray):
        _ = self.iterate_through_channel()
        if type == DataType.spectrum:
            x = self.x[7]  # dirty
            y = self.x[4]  # clean
        elif type == DataType.iq_samples:
            x = self.x[9]
            y = self.x[1]
        else:
            raise ValueError(f'unknown type: {type}, choose one of DataType\'s values')

        return x, y

    def step0_gen_msg(self):
        self.x[0] = self.cb.generate_message(self.length_of_msg)
        if self.verbose:
            Visualizer.print_bits(self.x[0], self.sps, 'message before channel')

    def step1_modulate(self):
        self.x[1] = self.cb.modulate(self.x[0], self.modem)
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
            print(f'ber = {num_errors/self.length_of_msg} = {num_errors}/{self.length_of_msg}')

    def step2_over_sample(self):
        self.x[2] = self.cb.over_sample(self.x[1], self.over_sampling)
        if self.verbose:
            Visualizer.my_plot(np.real(self.x[2][0:50]), name='zero padded - i (real)', function='stem')
            print(f'vec length = {len(self.x[2])}, over_sampling period = {self.over_sampling}')

    def step3_pulse_shaping(self):
        self.x[3], self._N_rrc, self._h_rrc = self.cb.pulse_shape(self.x[2], self.roll_off, self.over_sampling,
                                                                  self.Ts)

        if self.verbose:
            zm = range(self.N_rrc//2, self.N_rrc//2 + 24)
            Visualizer.twin_zoom_plot('|h_rrc filter|^2', np.abs(self.h_rrc) ** 2, zm)
            Visualizer.twin_zoom_plot('real{X(xi) * h(xi)}', np.real(self.x[3]), range(0, self.N_rrc))

            print(f'filter len = {self.N_rrc}, signal len = {len(self.x[3])}')

        self._test_step3()

    def _test_step3(self):

        xx41 = np.convolve(self.x[3], self.h_rrc)

        # sampling the analog vector into discrete bits
        N_rrc = len(self.h_rrc)
        start = N_rrc
        stop = - N_rrc
        step = self.over_sampling
        xx42 = xx41[start:stop:step]/self.over_sampling
        x0_reconst = self.modem.demodulate(xx42)
        num_errors = (x0_reconst != self.x[0]).sum()

        assert num_errors == 0, \
            f"test pulse shaping failed, there are {num_errors} errors instead of 0"

        if self.test_verbose:
            Visualizer.twin_zoom_plot('analog signal after another conv', np.real(xx41), range(0, N_rrc*2))

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            # Visualizer.my_plot(np.real(xx41), name='analog signal after another conv', ax=ax1, hold=1)
            # Visualizer.my_plot(np.real(xx41[:N_rrc * 2]), name='zoom in', ax=ax2)

            Visualizer.my_plot(np.real(xx42[0:50]), name='sampled bits', function='stem')

            Visualizer.plot_constellation_map_with_points(xx42, self.m_qam, 'after depulse shaping')
            Visualizer.eye_diagram(xx42, sps=self.sps)

    def step4_pre_equalize(self):
        self.N_xi, self.N_time, self.tvec, self.xivec, self.BW = self.cb.gen_nft_params(len(self.x[3]), self.dt)

        if self.verbose:
            print(
                f'xi ∈ [{self.xivec[0]/1e9:.2f}:{self.xivec[-1]/1e9:.2f}] GHz ,\t N_xi   (=M) = {self.N_xi}\n'
                f't  ∈ [{self.tvec[0]*1e12:.2f}:{self.tvec[-1]*1e12:.2f}] ps    ,\t N_time (=D) = {self.N_time}\n'
                f'BW = {self.BW/1e9:.2f} GHz'
            )

        self.x[4] = self.cb.pre_equalize(self.x[3], self.normalization_factor)

        if self.verbose:
            Visualizer.twin_zoom_plot('|X(xi)|', np.abs(self.x[4]), range(4000, 4200), self.xivec, xlabel='xi')
            print(f'signal len = {len(self.x[4])}')

    def step5_inft(self):
        self.x[5] = self.cb.inft(self.x[4], self.tvec, self.xivec)  # , self.P_0)

        if self.verbose:
            print(f'length of INFT(x) = {len(self.x[5])}')
            Visualizer.print_signal_specs(self.x[5], self.tvec)
            Visualizer.twin_zoom_plot('real{X(xi)}', np.real(self.x[5]), range(4000, 4200), self.tvec, 't')

    def step6_channel(self):
        self.x[6] = self.cb.channel(self.x[5], self.ssf)  # , self.P_0)

        if self.verbose:
            Visualizer.twin_zoom_plot('real X(xi)', np.real(self.x[6]), range(4000, 4200))

    def step7_nft(self):
        self.x[7] = self.cb.nft(self.x[6], self.tvec, self.BW, self.N_xi)

        if self.verbose:
            zm = range(0, self.N_rrc)
            Visualizer.twin_zoom_plot('real {X(xi)}', np.real(self.x[7]), zm, self.xivec, 'xi[Hz]')
            if self.x[4] is not None:
                Visualizer.twin_zoom_plot('ref before INFT', np.real(self.x[4]), zm, self.xivec, 'xi[Hz]')

    def step8_equalize(self):
        self.x[8] = self.cb.equalizer(self.x[7], self.normalization_factor)

    def step9_match_filter(self):
        self.x[9], y1 = self.cb.match_filter(self.x[8], self.h_rrc, self.N_rrc, self.over_sampling)
        if self.verbose:
            Visualizer.twin_zoom_plot('analog signal after another conv (real)', np.real(y1),
                                      range(0, 2*self.N_rrc))
            Visualizer.twin_zoom_plot('sampled bits (real)', np.real(self.x[9]), range(0, 50), function='stem')

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
