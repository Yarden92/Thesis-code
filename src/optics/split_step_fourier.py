import os
import warnings

import numpy as np

from src.general_methods.visualizer import Visualizer
from src.general_methods.signal_processing import SP


class SplitStepFourier:
    def __init__(self,
                 b2=-20e-27,  # TODO: change to nano
                 gamma=0.003,
                 t0=125e-12,
                 dt=1,
                 z_n=1000e3,
                 dz=200,
                 with_noise=True,
                 verbose=False
                 ):

        Z_0 = t0 ** 2 / abs(b2)
        self.P_0 = 1/(gamma*Z_0)

        self.b2 = b2
        self.gamma = gamma
        self.t0 = t0
        self.dt = dt
        self.z_n = z_n
        self.dz = dz

        self.D = self._calculate_D()
        self.noise_amplitude = np.sqrt(self.D * self.dz / self.dt)
        self.with_noise = with_noise

        # self.N = np.int64((1 + z_n*(t0 ** 2)/np.absolute(b2))//self.h)
        self.N = int(z_n / self.dz)
        if verbose:
            print(f'SSF params: N = {self.N}')

        if self.N < 1:
            warnings.warn(f"there are not enough ({self.N}) steps in split algo do at least one of the following: \n"
                          f"\t1) reduce dz\n"
                          f"\t2) enlarge z_n\n")

    def __call__(self, q: np.ndarray) -> np.ndarray:
        u = q*np.sqrt(self.P_0)

        Nt = np.max(u.shape)  # length

        w = 2.0 * np.pi / (float(Nt) * self.dt) * np.fft.fftshift(np.arange(-Nt / 2.0, Nt / 2.0))

        # Half-step linear propagation
        half_step = np.exp((1j * self.b2 / 2.0 * w ** 2) * self.dz / 2.0)

        u = np.fft.fft(u)
        for _ in range(self.N):
            # First half-step linear propagation
            u = np.fft.ifft(u * half_step)

            # Noise addition
            u += self._get_noise(Nt)

            # Nonlinear propagation
            u *= np.exp(1j * self.gamma * self.dz * np.abs(u) ** 2)

            # Second half-step linear propagation
            u = half_step * np.fft.fft(u)

        u = np.fft.ifft(u)
        u /= np.sqrt(self.P_0)

        return u

    def params_to_dict(self):
        return {
            'b2': self.b2,
            'gamma': self.gamma,
            't0': self.t0,
            'dt': self.dt,
            'z_n': self.z_n,
            'dz': self.dz,
        }

    @classmethod
    def from_dict(cls, _dict: dict, verbose=False):
        return cls(
            b2=_dict['b2'],
            gamma=_dict['gamma'],
            t0=_dict['t0'],
            dt=_dict['dt'],
            z_n=_dict['z_n'],
            dz=_dict['dz'],
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
            dz=1000
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
        noise = (np.random.randn(L) + 1j * np.random.randn(L))
        noise = self.noise_amplitude * noise

        return noise

    def _calculate_D(self):
        h = 6.62607015e-34  # planck constant
        lambda_0 = 1.55 * 1e-6  # wavelength
        C = 299792458  # speed of light
        K_T = 1.1
        X_dB = 0.2  # dB/km -> TODO: do we need to do something with the km?
        # X = 10 ** (X_dB / 10) # fiber loss coefficient
        # X = (X_dB / 10) * np.log(10) * 1e-3  # fiber loss coefficient
        X = (X_dB / 10) * np.log(10) # fiber loss coefficient [km]
        v_0 = C / lambda_0  # frequency 

        D = 0.5 * h * v_0 * K_T * X  # D= 7.38e-20
        D = D * 1e12 # D= 7.38e-8 [v*ps/km]
        return D


def tester():
    ssf = SplitStepFourier(
        b2=-20e-27,
        gamma=0.003,
        t0=125e-12,
        dt=1,
        z_n=1000e3,
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
