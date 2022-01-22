import warnings

import numpy as np

from src.visualizer import Visualizer


class SplitStepFourier:
    def __init__(self,
                 alpha=0,
                 b2=-20e-27,
                 gamma=0.003,
                 t0=125e-12,
                 last_step=1.51e-3,
                 dt=1e-12,
                 h=10000
                 ):
        self.gamma = gamma
        self.b2 = b2
        self.dt = dt
        self.h = h

        self.alph = alpha / (4.343)
        self.N = np.int64((1 + last_step * (t0 ** 2) / np.absolute(b2)) // self.h)

        if self.N < 1:
            warnings.warn(f"there are not enough ({self.N}) steps in split algo do at least one of the following: \n"
                          f"\t1) reduce h\n"
                          f"\t2) enlarge t0\n"
                          f"\t3) enlarge last step\n")

    @classmethod
    def with_params_for_example(cls):
        return cls(
            alpha=0,
            b2=-20e-27,
            gamma=0.003,
            t0=125e-12,
            last_step=1.51,
            # steps=np.arange(0.1, 1.51, 0.1),
            dt=1e-12,
            h=1000
        )

    def start_minimal(self, x: np.ndarray) -> np.ndarray:
        Nt = np.max(x.shape)

        dw = 2.0 * np.pi / float(Nt) / self.dt

        w = dw * np.fft.fftshift(np.arange(-Nt / 2.0, Nt / 2.0))

        vec1 = np.exp((-self.alph + 1j * self.b2 / 2.0 * w ** 2) * self.h / 2.0)

        f = x
        for _ in range(self.N):
            f = np.fft.fft(f) * vec1
            f = np.fft.ifft(f * vec1)
            f *= np.exp(1j * self.h * self.gamma * np.absolute(f) ** 2)

        return f

    def __call__(self, x) -> np.ndarray:
        return self.start_minimal(x)

    def plot_input(self, t, x) -> None:
        Visualizer.my_plot(t, np.abs(x), name='input pulse |x(t)|', xlabel='time', ylabel='amplitude')

    def plot_output(self, t, y) -> None:
        Visualizer.my_plot(t, np.abs(y), name='output |y(y)|', xlabel='time')


def tester():
    ssf = SplitStepFourier(
        alpha=0,
        b2=-20e-27,
        gamma=0.003,
        t0=125e-12,
        last_step=1.51,
        # steps=np.arange(0.1, 1.51, 0.1),
        dt=1e-12,
        h=100
    )
    Po = .00064
    C = -2
    Ao = np.sqrt(Po)
    to = 125e-12
    tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)

    x = Ao * np.exp(-((1 + 1j * (-C)) / 2.0) * (tau_vec / to) ** 2)
    x = np.array(x)

    y = ssf(x)

    ssf.plot_input(tau_vec, x)
    ssf.plot_output(tau_vec, y)


if __name__ == '__main__':
    tester()
