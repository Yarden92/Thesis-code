import os
from typing import List
import warnings

import numpy as np

from src.general_methods.visualizer import Visualizer
from src.general_methods.signal_processing import SP


class SplitStepFourier:
    def __init__(self,
                 b2=-20e-27,  # TODO: change to nano
                 gamma=0.003,
                 dt=1,
                 L=1000e3,
                 Nt=4096,
                 dz=200,
                 K_T=1.1,
                 chi=0.0461,
                 with_noise=True,
                 verbose=False
                 ):

        self.b2 = b2
        self.gamma = gamma
        self.dt = dt
        self.L = L
        self.Nt = Nt
        self.dz = dz
        self.N = int(self.L / self.dz)

        self.K_T = K_T
        self.chi = chi

        self.D = self._calculate_D()
        self.noise_amplitude = np.sqrt(self.D * self.dz / self.dt)
        # print(f'dz={self.dz:.2e}, dt={self.dt:.2e}, D={self.D:.2e}')
        # print(f'noise amplitude: {self.noise_amplitude:.2e}')
        self.with_noise = with_noise
        self.noise_vecs = self._gen_noise_vecs()

        self.w = self._gen_w_axis()
        self.half_step = np.exp((1j * self.b2 / 2.0 * self.w ** 2) * self.dz / 2.0)
        self.full_step = np.exp((1j * self.b2 / 2.0 * self.w ** 2) * self.dz)


        if verbose:
            print(f'SSF params: N = {self.N}')

        if self.N < 1:
            warnings.warn(f"there are not enough ({self.N}) steps in split algo do at least one of the following: \n"
                          f"\t1) reduce dz\n"
                          f"\t2) enlarge L\n")

    def __call__(self, q: np.ndarray) -> np.ndarray:
        u = q        

        u = np.fft.ifft(np.fft.fft(u) * self.half_step)     # Half linear step

        for i in range(self.N-1):
            u *= self._get_nonlinear_step(u)                        # Nonlinear step
            u += self.noise_vecs[i]                                 # Add noise
            u = np.fft.ifft(np.fft.fft(u) * self.full_step)         # Full linear step

        u *= self._get_nonlinear_step(u)                    # Nonlinear step
        u += self.noise_vecs[self.N-1]                      # Add noise
        u = np.fft.ifft(np.fft.fft(u) * self.half_step)     # Half linear step

        return u

    def params_to_dict(self):
        return {
            'b2': self.b2,
            'gamma': self.gamma,
            'dt': self.dt,
            'L': self.L,
            'dz': self.dz,
        }

    @classmethod
    def from_dict(cls, _dict: dict, verbose=False):
        return cls(
            b2=_dict['b2'],
            gamma=_dict['gamma'],
            dt=_dict['dt'],
            L=_dict['L'],
            dz=_dict['dz'],
            verbose=verbose
        )

    @classmethod
    def with_params_for_example(cls):
        return cls(
            b2=-20e-27,
            gamma=0.003,
            L=1.51,
            # steps=np.arange(0.1, 1.51, 0.1),
            # dt=1e-12,
            dz=1000
        )

    def plot_input(self, t, x) -> None:
        Visualizer.my_plot(t, np.abs(x), name='input pulse |x(t)|', xlabel='time', ylabel='amplitude')

    def plot_output(self, t, y) -> None:
        Visualizer.my_plot(t, np.abs(y), name='output |y(y)|', xlabel='time')

    def _get_nonlinear_step(self, u: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.gamma * self.dz * np.abs(u) ** 2)

    def _get_noise(self, L) -> np.ndarray:
        if self.with_noise:
            return self._gen_noise(L)
        else:
            return self._get_dummy_noise(L)

    def _get_dummy_noise(self, L) -> np.ndarray:
        return np.zeros(L) + 1j*np.zeros(L)

    def _gen_noise(self, L) -> np.ndarray:
        noise = (np.random.randn(L) + 1j * np.random.randn(L))
        noise = self.noise_amplitude * noise

        return noise

    def _calculate_D(self):
        h = 6.62607015e-34  # planck constant [J*s]
        lambda_0 = 1.55 * 1e-6  # wavelength [m]
        C = 299792458  # speed of light [m/s]
        # # K_T = 1.1   # [unitless]
        # # X_dBkm = 0.2  # fiber loss coefficient [dB/km]
        # # X_km = 10 ** (X_dBkm / 10) # [1/km]
        # X = X_km * 1e-3  #  [1/m]
        K_T = self.K_T
        X = self.chi
        nu_0 = C / lambda_0  # frequency [Hz]

        D_Jm = 0.5 * h * nu_0 * K_T * X  # [J/m=Ws/km]
        normalizer = 1e12  # ps->s 
        D = D_Jm * normalizer  # [W*ps/km]

        return D
    
    def _gen_noise_vecs(self) -> List[np.ndarray]:
        noise_vecs = []
        Tb = 10240
        for _ in range(self.N):
            noise_vecs.append(self._get_noise(self.Nt))
            # print(f'average noise power: {SP.signal_power_dbm(noise_vecs[-1],self.dt,Tb)} dBm')

        return noise_vecs
    
    def _gen_w_axis(self) -> np.ndarray:
        w = 2.0 * np.pi / (self.Nt * self.dt) * np.fft.fftshift(np.arange(-self.Nt / 2.0, self.Nt / 2.0))
        return w


def tester():
    ssf = SplitStepFourier(
        b2=-20e-27,
        gamma=0.003,
        dt=1,
        L=1000e3,
        dz=200,
        D=1e30,
        with_noise=True,
        verbose=True
    )

    Po = .00064
    C = -2
    Ao = np.sqrt(Po)
    to = 125e-12
    tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)

    x = Ao*np.exp(-((1 + 1j*(-C))/2.0)*(tau_vec/to) ** 2)
    x = np.array(x)

    y = ssf(x)

    Visualizer.my_plot(tau_vec, np.abs(x), tau_vec, np.abs(y), name='output |y(y)|', xlabel='time',
                       legend=['in', 'out'])


if __name__ == '__main__':
    tester()
