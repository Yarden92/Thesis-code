import numpy as np
from ModulationPy import ModulationPy
from commpy.filters import rrcosfilter

from src.optics.myFNFTpy.FNFTpy import nsev_inverse_xi_wrapper, nsev_inverse, nsev
# TODO: take out prints into channel simulator, rather than channel blocks
from src.optics.split_step_fourier import SplitStepFourier


class ChannelBlocks:

    def generate_message(self, length_of_msg):  # step0_gen_msg
        y = np.random.choice([0, 1], size=length_of_msg)  # generate random message - binary vec [0,1,0,...]
        return y

    def generate_modem(self, m_qam):
        return ModulationPy.QAMModem(m_qam, soft_decision=False)

    def modulate(self, x, modem: ModulationPy.QAMModem):  # step 1
        y = modem.modulate(x)  # r[xi,0] = [1+j,-1+j,1-j,...]
        return y

    def over_sample(self, x, over_sampling):  # step 2
        y = np.zeros(over_sampling * len(x), dtype=np.complex64)
        y[::over_sampling] = x
        return y

    def pulse_shape(self, x, roll_off, over_sampling, Ts):  # step 3
        # design the filter length to complete x to be power of 2
        N_rrc, h_rrc = self.gen_h_rrc(len(x), roll_off, over_sampling, Ts)
        y = np.convolve(h_rrc, x)  # Waveform with PSF

        return y, N_rrc, h_rrc

    def gen_h_rrc(self, len_x, roll_off, over_sampling, Ts):
        desired_len = int(2 ** np.ceil(np.log2(len_x)))  # the next power of 2
        if desired_len - len_x < 100:
            desired_len = int(2 ** (1 + np.ceil(np.log2(len_x))))  # the next next power of 2

        N_rrc = int(np.ceil(desired_len - len_x)) + 1
        alpha: float = roll_off  # default = 0.25
        fs = over_sampling / Ts
        h_ind, h_rrc = rrcosfilter(N_rrc, alpha, Ts, fs)

        return N_rrc, h_rrc

    def pre_equalize(self, x, normalization_factor):  # step 4
        # normalize vector:
        y = normalization_factor * x

        return y

    def gen_nft_params(self, N_xi, dt):  # step 5.1
        # some basic params for NFT
        # N_xi = len(x)  # (=M)
        N_time = int(2 ** np.floor(np.log2(N_xi)))  # (=D)
        tvec = np.arange(start=-N_time / 2, stop=N_time / 2) * dt  # np.linspace(-t0, t0, N_time)
        rv, xi = nsev_inverse_xi_wrapper(N_time, tvec[0], tvec[-1], N_xi, display_c_msg=True)
        xivec = xi[0] + np.arange(N_xi) * (xi[1] - xi[0]) / (N_xi - 1)
        BW = xivec.max()
        # dt = tvec[1]-tvec[0]

        return N_xi, N_time, tvec, xivec, BW

    def inft(self, x, tvec, xivec):  # , P_0):  # step 5.2
        # INFT
        contspec = x
        bound_states = []  # np.array([0.7j, 1.7j])
        discspec = []  # [1.0, -1.0]

        cst = 1  # continuous spectrum type - default is None
        dst = 0  # default is None

        res = nsev_inverse(xivec, tvec, contspec, bound_states, discspec,
                           cst=cst, dst=dst, display_c_msg=False)

        assert res['return_value'] == 0, "INFT failed"

        x1 = res['q']  # q[t,0]
        return x1

    def channel(self, x, ssf: SplitStepFourier):  # , P_0):  # step 6
        x1 = ssf(x)
        return x1

    def nft(self, x, tvec, BW, N_xi):  # step 7
        res = nsev(x, tvec, Xi1=-BW, Xi2=BW, M=N_xi, display_c_msg=False)
        assert res['return_value'] == 0, "NFT failed"
        y = res['cont_ref']  # r[xi,L]

        return y

    def equalizer(self, x, normalization_factor):  # step 8
        # unnormalize
        y1 = x / normalization_factor

        # channel equalizer (nothing)
        equalizer_func = 1
        y2 = y1 * equalizer_func
        return y2

    def match_filter(self, x, h_rrc, N_rrc, over_sampling):  # step 9
        y1 = np.convolve(x, h_rrc)

        # sampling the analog vector into discrete bits
        start = N_rrc
        stop = - N_rrc  # + over_sampling
        step = over_sampling
        y2 = y1[start:stop:step] / over_sampling

        return y2, y1

    def demodulate(self, x, modem, length_of_msg):  # step 10
        y = modem.demodulate(x)
        assert len(y) == length_of_msg, \
            f"oh no, the outcome is not {length_of_msg}, but {len(y)}"
        return y

    def calc_ber(self, x_in: np.ndarray, x_out: np.ndarray, length_of_msg, sps):
        num_errors = (x_in != x_out).sum()
        ber = num_errors / length_of_msg

        return ber, num_errors
