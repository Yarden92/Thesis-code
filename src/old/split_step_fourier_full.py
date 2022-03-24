import numpy as np
from matplotlib import pyplot as plt, cm
from tqdm import tqdm

from src.general_methods.visualizer import Visualizer


# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.


class SplitStepFourier:
    def __init__(self,
                 Po=.00064,
                 alpha=0,
                 gamma=0.003,
                 t0=125e-12,
                 C=-2,
                 b2=-20e-27,
                 dt=1e-12,
                 rel_error=1e-5,
                 h=1000
                 ):
        # self.Po = Po
        self.alph = alpha / (4.343)
        self.gamma = gamma
        # self.to = to
        # self.C = C
        self.b2 = b2
        self.Ld = (to ** 2) / np.absolute(b2)
        # self.Ao = np.sqrt(Po),
        # self.tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)
        self.steps = np.arange(0.1, 1.51, 0.1)
        self.dt = dt
        # self.rel_error = rel_error
        self.h = h

        N = len(self.steps)
        Ntau = len(self.tau_vec)

        self.op_pulse = [[0] * Ntau] * N
        self.pbratio = [0] * N
        self.phadisp = [0] * N

        self.is_ready = False
        self.output = None

    def start_minimal(self, uaux: np.ndarray):
        # assert self.is_ready == False, "you already started!"
        if self.is_ready:
            self.op_pulse = []
            self.output = None
            self.is_ready = False

        l = np.max(uaux.shape)
        spectrum_aux = np.fft.fft(uaux)

        for i, step in enumerate(tqdm(self.steps)):
            z = step * self.Ld
            dw = 1.0 / float(l) / self.dt * 2.0 * np.pi

            w = dw * np.arange(-1 * l / 2.0, l / 2.0, 1)
            w = np.asarray(w)
            w = np.fft.fftshift(w)

            spectrum = spectrum_aux
            vec1 = np.exp((-self.alph + 1j * self.b2 / 2.0 * w ** 2) * self.h / 2.0)
            for _ in np.arange((z + 1) // self.h):
                f = np.fft.ifft(spectrum * vec1)
                f = f * np.exp(1j * self.h * self.gamma * np.absolute(f) ** 2)
                spectrum = np.fft.fft(f) * vec1

            f = np.fft.ifft(spectrum)

            self.op_pulse[i] = np.absolute(f)

        self.output = self.op_pulse[-1]
        self.is_ready = True

    def start_fully(self, uaux):
        assert self.is_ready == False, "you already started!"

        np_uaux = np.asarray(uaux)
        l = np.max(uaux.shape)
        spectrum_aux = np.fft.fft(np_uaux)

        for i, ii in enumerate(tqdm(self.steps)):
            z = ii * self.Ld
            fwhml = np.nonzero(np.absolute(uaux) > np.absolute(np.max(np.real(uaux)) / 2.0))
            fwhml = len(fwhml[0])
            dw = 1.0 / float(l) / self.dt * 2.0 * np.pi

            w = dw * np.arange(-1 * l / 2.0, l / 2.0, 1)
            w = np.asarray(w)
            w = np.fft.fftshift(w)

            spectrum = spectrum_aux
            vec1 = np.exp((-self.alph + 1j * self.b2 / 2.0 * w ** 2) * self.h / 2.0)
            for jj in np.arange((z + 1) // self.h):
                f = np.fft.ifft(spectrum * vec1)
                f = f * np.exp(1j * self.h * self.gamma * np.absolute(f) ** 2)
                spectrum = np.fft.fft(f) * vec1

            f = np.fft.ifft(spectrum)
            fwhm = np.nonzero(np.absolute(f) > np.absolute(np.max(np.real(f)) / 2.0))
            fwhm = len(fwhm[0])
            ratio = float(fwhm) / fwhml
            im = np.absolute(np.imag(f))
            re = np.absolute(np.real(f))
            div = np.dot(im, np.linalg.pinv([re]))
            dd = np.degrees(np.arctan(div))

            self.op_pulse[i] = np.absolute(f)
            self.pbratio[i] = ratio
            self.phadisp[i] = dd[0]
            # print("> %d / 14 " % ln)

        self.output = self.op_pulse[-1]
        self.is_ready = True

    def __call__(self, x):
        self.start_minimal(x)
        return self.output

    def plot_input(self, uaux):
        # uaux
        Visualizer.my_plot(np.abs(uaux), name='input pulse', xlabel='time', ylabel='amplitude')

    def plot_stuffs(self):
        assert self.is_ready, "please execute .start(uaux) before calling prints"
        # phadisp
        Visualizer.my_plot(self.phadisp, name='phase change')
        # pbratio
        Visualizer.my_plot(self.pbratio, name='pulse broadening')

    def plot_3d(self):
        assert self.is_ready, "please execute .start(uaux) before calling prints"

        # op_pulse
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        Z = np.asarray(self.op_pulse)
        Ny, Nx = Z.shape
        X, Y = np.arange(Nx), np.arange(Ny)  # time
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap('coolwarm'))
        plt.title('Pulse evolution , z=amplitude')
        plt.xlabel('Time')
        plt.ylabel('Distance')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_output(self):
        Visualizer.my_plot(self.output, name='output', xlabel='time')


# units are SI


# def plots(folder, ln, op_pulse, pbratio, phadisp, uaux):
#     print("\n\n> Plotting...")
#     trace_pulse_broad = go.Scatter(y=pbratio[0:ln], x=np.arange(1, ln + 1, 1))
#     trace_phase_change = go.Scatter(y=phadisp[0:ln], x=np.arange(1, ln + 1, 1))
#     layout_input_pulse = go.Layout(
#         autosize=False,
#         width=500,
#         height=400,
#         title='Input Pulse',
#         xaxis=dict(
#             title='Time',
#             titlefont=dict(
#                 family='Courier New, monospace',
#                 size=18,
#                 color='#7f7f7f'
#             )
#         ),
#         yaxis=dict(
#             title='Amplitude',
#             titlefont=dict(
#                 family='Courier New, monospace',
#                 size=18,
#                 color='#7f7f7f'
#             )
#         )
#     )
#
#     trace_input_pulse = go.Scatter(y=np.absolute(uaux))
#     input_pulse = go.Figure(data=[trace_input_pulse], layout=layout_input_pulse)
#     Path(folder).mkdir(parents=True, exist_ok=True)
#
#     plot([trace_phase_change], filename=f'./{folder}/phase_change.html')
#     plot([trace_pulse_broad], filename=f'./{folder}/pulse_broadening.html')
#     plot(input_pulse, filename=f'./{folder}/input_pulse.html')
#
#     trace_pulse_evolution = go.Surface(z=op_pulse, colorscale='Jet')
#     layout_pulse_evolution = go.Layout(
#         autosize=False,
#         width=800,
#         height=800,
#         title='Pulse Evolution',
#         scene=go.Scene(
#             xaxis=go.XAxis(title='Time'),
#             yaxis=go.YAxis(title='Distance'),
#             zaxis=go.ZAxis(title='Amplitude'))
#     )
#     pulse_evolution = go.Figure(data=[trace_pulse_evolution], layout=layout_pulse_evolution)
#     plot(pulse_evolution, filename=f'./{folder}/pulse_evolution.html')

def split_channel(x):
    ssf = SplitStepFourier()
    y = ssf.start_minimal(x)
    return x


def tester():
    ssf = SplitStepFourier()

    uaux = ssf.Ao * np.exp(-((1 + 1j * (-ssf.C)) / 2.0) * (ssf.tau_vec / ssf.to) ** 2)
    uaux = np.array(uaux)

    ssf.start_minimal(uaux)

    ssf.plot_input(uaux)
    ssf.plot_output()


if __name__ == '__main__':
    tester()
