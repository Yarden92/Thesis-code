import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upsample, convolve
from scipy.fft import fft, ifft
from scipy.special import erfc

class NFDMModulator:
    def __init__(self):
        # Initialize default system parameters
        self.Nbursts = 1
        self.Nspans = 12
        self.La = 12 * 80 / self.Nspans
        self.beta2 = -21.0
        self.gamma = 1.27
        self.eta = 2
        self.W = 0.05
        self.TG = 1.5 * np.pi * self.W * abs(self.beta2) * self.La * self.Nspans
        self.Nsc = round(self.W * self.TG / (self.eta - 1))
        self.Nsc = 2 * (round(self.Nsc / 2))
        self.T0 = self.Nsc / self.W
        self.Tb = self.T0 * self.eta
        self.no_ss = True
        self.noise = False
        self.Nghost = 2
        self.lumped = True
        self.dz = 0.2
        self.Nsps = np.ceil(self.La / self.dz)
        self.dz = self.La / self.Nsps
        self.L = self.La * self.Nspans
        self.alphadB = 0.2
        self.mu = 0.1
        self.M = 32
        self.bet = 0.2
        self.Nos = 16

        self.set_time_and_frequency_arrays()

    def set_time_and_frequency_arrays(self):
        # Setting the time and (nonlinear) frequency arrays for each burst
        self.Ns = (self.Nsc + 2 * self.Nghost) * self.Nos
        self.t = np.arange(self.Ns) / self.Ns
        self.xi = np.arange(-self.Ns // 2, self.Ns // 2) / self.Nos

        # self.xi = np.arange(-self.Ns / 2, self.Ns / 2) / self.Ns
        # self.xi = np.fft.fftshift(self.xi)
        # self.xi = self.xi * self.Ns / self.Tb

    def rrc_impulse(self, t, Ts, bet):
        temp = bet * np.cos((1 + bet) * np.pi * t / Ts)
        temp += (np.pi * (1 - bet) / 4) * np.sinc((1 - bet) * t / Ts)
        temp /= (1 - (4 * bet * t / Ts) ** 2)
        Psi = temp * 4 / (np.pi * np.sqrt(Ts))
        poles = np.isinf(temp)
        Psi[poles] = (bet / np.sqrt(Ts)) * (
                    -(2 / np.pi) * np.cos(np.pi * (1 + bet) / (4 * bet)) + np.sin(np.pi * (1 + bet) / (4 * bet)))
        return Psi

    def generate_bursts(self):
        print('Generating bursts')
        dataMod = np.zeros((self.Nbursts, self.Nsc), dtype=complex)
        uin = np.zeros((self.Nbursts, self.Ns), dtype=complex)
        psi_xi = self.rrc_impulse(self.xi, 1, self.bet)

        for ii in range(self.Nbursts):
            message = np.random.randint(0, self.M, self.Nsc + 2 * self.Nghost)
            c = qammod(message, M, 'gray')
            dataMod[ii, :] = np.concatenate([c[:Nsc // 2], c[Nsc // 2 + 2 * Nghost:]])
            c[Nsc // 2 + 1:Nsc // 2 + 2 * Nghost] = 0
            c = upsample(c, Nos)
            uin[ii, :] = mu * ifft(fft(psi_xi) * fft(c))

        print('Done')


    def execute(self):
        self.generate_bursts()
        self.inverse_nft_tx()
        self.split_step_propagation()
        self.forward_nft_rx()
        self.demodulation_data_processing()
        self.plot_constellation_diagram()


if __name__ == "__main__":
    modulator = NFDMModulator()
    modulator.execute()
