import numpy as np

from src.visualizer import Visualizer


class SplitStepFourier:
    def __init__(self,
                 alpha=0,
                 b2=-20e-27,
                 gamma=0.003,
                 to=125e-12,
                 steps=np.arange(0.1, 1.51, 0.1),
                 dt=1e-12,
                 h=1000
                 ):
        self.gamma = gamma
        self.b2 = b2
        self.steps = steps
        self.dt = dt
        self.h = h

        self.alph = alpha / (4.343)
        self.Ld = (to ** 2) / np.absolute(b2)

    def start_minimal(self, x: np.ndarray) -> np.ndarray:

        l = np.max(x.shape)
        spectrum_aux = np.fft.fft(x)

        dw = 1.0 / float(l) / self.dt * 2.0 * np.pi

        w = dw * np.arange(-1 * l / 2.0, l / 2.0, 1)
        w = np.asarray(w)
        w = np.fft.fftshift(w)

        vec1 = np.exp((-self.alph + 1j * self.b2 / 2.0 * w ** 2) * self.h / 2.0)

        for i, step in enumerate(self.steps):
            z = step * self.Ld

            spectrum_i = spectrum_aux
            for _ in np.arange((z + 1) // self.h):
                f = np.fft.ifft(spectrum_i * vec1)
                f = f * np.exp(1j * self.h * self.gamma * np.absolute(f) ** 2)
                spectrum_i = np.fft.fft(f) * vec1

        y = np.absolute(f)
        return y

    def __call__(self, x) -> np.ndarray:
        return self.start_minimal(x)

    def plot_input(self, x):
        Visualizer.my_plot(np.abs(x), name='input pulse', xlabel='time', ylabel='amplitude')

    def plot_output(self, y):
        Visualizer.my_plot(y, name='output', xlabel='time')


def tester():
    ssf = SplitStepFourier(
        alpha=0,
        b2=-20e-27,
        gamma=0.003,
        to=125e-12,
        steps=np.arange(0.1, 1.51, 0.1),
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

    ssf.plot_input(x)
    ssf.plot_output(y)


if __name__ == '__main__':
    tester()
