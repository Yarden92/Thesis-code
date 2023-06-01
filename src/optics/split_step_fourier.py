import os
import warnings

import numpy as np

from src.general_methods.visualizer import Visualizer
from src.general_methods.signal_processing import SP


class SplitStepFourier:
    def __init__(self,
                 b2=-20e-27,
                 gamma=0.003,
                 t0=125e-12,
                 dt=1,
                 z_n=1000e3,
                 h=200,
                 noise_psd=1e-3,
                 with_noise=True,
                 verbose=False
                 ):
        self.gamma = gamma
        self.b2 = b2
        self.h = h
        self.t0 = t0
        self.z_n = z_n

        

        Z_0 = t0 ** 2/abs(b2)
        self.P_0 = 1/(gamma*Z_0)

        self.dt = dt


        self.noise_psd = noise_psd
        self.with_noise = with_noise
        self.noise_mean = 0
        self.std = np.sqrt(self.noise_psd * self.h / self.dt)

        # self.N = np.int64((1 + z_n*(t0 ** 2)/np.absolute(b2))//self.h)
        self.N = int(z_n / self.h)
        if verbose:
            print(f'SSF params: N = {self.N}, P_0 = {self.P_0}')

        if self.N < 1:
            warnings.warn(f"there are not enough ({self.N}) steps in split algo do at least one of the following: \n"
                          f"\t1) reduce h\n"
                          f"\t2) enlarge t0\n"
                          f"\t3) enlarge z_n\n")

        

    def __call__(self, x) -> np.ndarray:
        L = len(x)
        x = x*np.sqrt(self.P_0)

        Nt = np.max(x.shape)

        dw = 2.0*np.pi/float(Nt)/self.dt

        w = dw*np.fft.fftshift(np.arange(-Nt/2.0, Nt/2.0))

        vec1 = np.exp((1j*self.b2/2.0*w ** 2)*self.h)

        a = x
        for _ in range(self.N):
            a = np.fft.fft(a)*vec1
            a = np.fft.ifft(a)
            a *= np.exp(1j*self.h*self.gamma*np.abs(a) ** 2)
            a += self._get_noise(L)
        a = a/np.sqrt(self.P_0)

        return a

    def params_to_dict(self):
        return {
            'b2': self.b2,
            'gamma': self.gamma,
            't0': self.t0,
            'dt': self.dt,
            'z_n': self.z_n,
            'h': self.h,
        }

    @classmethod
    def from_dict(cls, _dict: dict, verbose=False):
        return cls(
            b2=_dict['b2'],
            gamma=_dict['gamma'],
            t0=_dict['t0'],
            dt=_dict['dt'],
            z_n=_dict['z_n'],
            h=_dict['h'],
            verbose=verbose
        )
        
    @classmethod
    def with_params_for_example(cls):
        return cls(
            b2=-20e-27,
            gamma=0.003,
            t0=125e-12,
            z_n=1.51,
            # steps=np.arange(0.1, 1.51, 0.1),
            # dt=1e-12,
            h=1000
        )

    # def set_dt(self, dt):
    #     self.dt = dt

    def plot_input(self, t, x) -> None:
        Visualizer.my_plot(t, np.abs(x), name='input pulse |x(t)|', xlabel='time', ylabel='amplitude')

    def plot_output(self, t, y) -> None:
        Visualizer.my_plot(t, np.abs(y), name='output |y(y)|', xlabel='time')

    def _get_noise(self, L) -> np.ndarray:
        if self.with_noise:
            return self._gen_noise(L)
        else:
            return self._get_dummy_noise(L)

    def _get_dummy_noise(self, L) -> np.ndarray:
        return np.zeros(L) + 1j*np.zeros(L)
    
    def _gen_noise(self, L) -> np.ndarray:
        real = np.random.normal(self.noise_mean, self.std, L)
        imag = np.random.normal(self.noise_mean, self.std, L)
        return real + 1j*imag


def tester():
    ssf = SplitStepFourier(
        b2=-20e-27,
        gamma=0.003,
        t0=125e-12,
        dt=1,
        z_n=1000e3,
        h=200
    )
    Po = .00064
    C = -2
    Ao = np.sqrt(Po)
    to = 125e-12
    tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)

    x = Ao*np.exp(-((1 + 1j*(-C))/2.0)*(tau_vec/to) ** 2)
    x = np.array(x)

    y = ssf(x)

    # ssf.plot_input(tau_vec, x)
    # ssf.plot_output(tau_vec, y)
    Visualizer.my_plot(tau_vec, np.abs(x), tau_vec, np.abs(y), name='output |y(y)|', xlabel='time',
                       legend=['in', 'out'])


# def test2_test_snrs():
#     # init input signal
#     Po = .00064
#     C = -2
#     Ao = np.sqrt(Po)
#     to = 125e-12
#     tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)
#     x = Ao*np.exp(-((1 + 1j*(-C))/2.0)*(tau_vec/to) ** 2)
#     x = np.array(x)

#     snr_list = [-10, -5, 0, 5, 10]
#     os.makedirs('snrs', exist_ok=True)

#     for snr in snr_list:
#         ssf = SplitStepFourier(
#             dt=1,
#             z_n=100e3,
#             h=200,
#             snr=snr
#         )
#         y = ssf(x)
#         Visualizer.my_plot(tau_vec, np.abs(x), tau_vec, np.abs(y),
#                            name=f'output |y(y)|, snr = {snr}',
#                            xlabel='time', legend=['in', 'out'],
#                            output_name=f'snrs/output_snr_{snr}.png')


if __name__ == '__main__':
    tester()
