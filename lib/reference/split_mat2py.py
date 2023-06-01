import numpy as np

class SplitMat2Py:
    def __init__(self, dt, dz, nz, beta2, gamma, D, display):
        # Constructor
        self.dt = dt
        self.dz = dz
        self.nz = nz
        self.beta2 = beta2
        self.gamma = gamma
        self.D = D
        self.display = display

    def __call__(self, x):
        # u0 = np.reshape(x, (1, len(x)))
        u0 = x
        nt = len(u0)
        w = 2 * np.pi * np.concatenate((np.arange(nt//2), np.arange(-nt//2, 0))) / (self.dt * nt)

        halfstep = 1j * self.beta2 * (w ** 2) / 2
        halfstep = np.exp(halfstep * self.dz / 2)

        ufft = np.fft.fft(u0)
        if self.display:
            if self.dz >= 0:
                print('Propagating. Please wait...')
            else:
                print('Back-propagating. Please wait...')

        for iz in range(self.nz):
            uhalf = np.fft.ifft(halfstep * ufft)
            noise = np.sqrt(self.D * self.dz / self.dt) * (np.random.randn(1, nt) + 1j * np.random.randn(1, nt))
            uhalf = uhalf + noise
            uv = uhalf * np.exp(1j * self.gamma * (np.abs(uhalf) ** 2) * self.dz)
            ufft = halfstep * np.fft.fft(uv)

            if self.display and iz % 4 == 0:
                progress = (iz + 1) / self.nz * 100
                print(f"Progress: {progress:.2f}%")

        if self.display:
            print('Propagation completed.')

        u1 = np.fft.ifft(ufft)

        return u1

Po = .00064
C = -2
Ao = np.sqrt(Po)
to = 125e-12
tau_vec = np.arange(-4096e-12, 4095e-12, 1e-12)

x = Ao*np.exp(-((1 + 1j*(-C))/2.0)*(tau_vec/to) ** 2)
x = np.array(x)


matpy_ssf = SplitMat2Py(
    dt=1,
    dz=200,
    nz=1000,
    beta2=-20e-27,
    gamma=0.003,
    D=0.1,
    display=False
)

y2 = matpy_ssf(x)

