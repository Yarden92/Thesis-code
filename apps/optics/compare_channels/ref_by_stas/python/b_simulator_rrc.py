import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, convolve
from scipy.fft import fft, ifft
from scipy.special import erfc

def main():
    # General system parameters
    Nbursts = 1
    Nspans = 12
    La = 12 * 80 / Nspans
    beta2 = -21.0
    gamma = 1.27
    eta = 2
    W = 0.05
    TG = 1.5 * np.pi * W * abs(beta2) * La * Nspans
    Nsc = round(W * TG / (eta - 1))
    Nsc = 2 * (round(Nsc / 2))
    T0 = Nsc / W
    Tb = T0 * eta
    no_ss = True
    noise = False
    Nghost = 2

    # Amplification scheme
    lumped = True
    if not lumped:
        Nspans = 1
    dz = 0.2
    Nsps = np.ceil(La / dz)
    dz = La / Nsps
    L = La * Nspans
    alphadB = 0.2

    # Linear loss
    if lumped:
        alpha = alphadB * np.log(10) / 10
    else:
        alpha = 0

    G = np.exp(alpha * La)
    gamma_eff = gamma
    if lumped and G > 1:
        gamma_eff = gamma * (G - 1) / (G * np.log(G))

    # Modulation constellation
    mu = 0.1
    M = 32

    # Carrier characteristics
    bet = 0.2

    # Normalized units
    Tn = T0 / (np.pi * (1 + bet))
    Zn = Tn**2 / abs(beta2)
    Pn = 1 / (gamma_eff * Zn)

    # Setting the time and (nonlinear) frequency arrays for each burst
    Nos = 16
    Ns = (Nsc + 2 * Nghost) * Nos
    Nnft = 4 * 2**(np.ceil(np.log2(Ns)))
    Tnft = np.pi * Nos * Tn
    dt = Tnft / Nnft
    Nb = 2 * round(Tb / (2 * dt))
    tau = dt * np.arange(-Nnft // 2, Nnft // 2) / Tn
    xi = np.arange(-Ns // 2, Ns // 2) / Nos

    # Creating symbols and bursts
    print('Generating bursts')
    dataMod = np.zeros((Nbursts, Nsc), dtype=complex)
    uin = np.zeros((Nbursts, Ns), dtype=complex)
    psi_xi = rrc_impulse(xi, 1, bet)

    for ii in range(Nbursts):
        message = np.random.randint(0, M, Nsc + 2 * Nghost)
        c = qammod(message, M, 'gray')
        dataMod[ii, :] = np.concatenate([c[:Nsc // 2], c[Nsc // 2 + 2 * Nghost:]])
        c[Nsc // 2 + 1:Nsc // 2 + 2 * Nghost] = 0
        c = upsample(c, Nos)
        uin[ii, :] = mu * ifft(fft(psi_xi) * fft(c))

    print('Done')

    # INFT @Tx
    Nb = min(Nb, Nnft)
    t = np.arange(-Nb * Nbursts // 2, Nb * Nbursts // 2) * dt
    q = np.zeros(Nbursts * Nb, dtype=complex)

    for ii in range(Nbursts):
        bin = np.sqrt(1 - np.exp(-np.abs(uin[ii, :])**2)) * np.exp(1j * np.angle(uin[ii, :]))
        bin *= np.exp(-1j * xi**2 * (L / Zn))
        bin_padded = np.zeros(Nnft, dtype=complex)
        bin_padded[Nnft // 2 - Ns // 2:Nnft // 2 + Ns // 2] = bin
        qin = mex_fnft_nsev_inverse(bin_padded, tau, Nnft, 'cstype_b_of_xi')
        qb = qin[(Nnft - Nb) // 2:(Nnft + Nb) // 2]
        left = 1 + Nb * (ii - 1)
        right = left + Nb - 1
        q[left:right] = qb

    q = q * np.sqrt(Pn)

    # Split-step (TBD)
    I0 = np.abs(q)**2 / 1e-3
    I1 = np.abs(qz)**2 / 1e-3

    plt.figure()
    plt.plot(t, I0)
    plt.hold(True)
    plt.plot(t, I1)
    for ii in range(Nbursts):
        plt.axvline((ii - 1/2 - (Nbursts - 1)/2) * Tb, color='red', linestyle='--')

    plt.xlabel('Time, ps')
    plt.ylabel('Power, mW')
    plt.legend(['Initial optical field', 'Final optical field'], loc='lower right')
    plt.xlim([-1, 1] * Tb * Nbursts)

    # Forward NFT @RX
    fig = plt.figure(figsize=(10, 5))
    uout = np.zeros((Nbursts, Ns), dtype=complex)
    qz /= np.sqrt(Pn)


    for ii in range(Nbursts):
        left = 1 + Nb * (ii - 1)
        right = left + Nb - 1
        qb = qz[left:right]
        q_padded = np.zeros(Nnft, dtype=complex)

        if Nnft > Nb:
            q_padded[Nnft // 2 - Nb // 2:Nnft // 2 + Nb // 2] = qb
        else:
            q_padded = qb[Nb // 2 - Nnft // 2:Nb // 2 + Nnft // 2]

        contspec = mex_fnft_nsev(q_padded, tau, Nnft, 'M', 'skip_bs', 'cstype_ab')
        bout_padded = contspec[Nnft:Nnft * 2]
        bout = bout_padded[Nnft // 2 - Ns // 2:Nnft // 2 + Ns // 2]

        if no_ss:
            bout *= np.exp(1j * xi**2 * (L / Zn))
        else:
            bout *= np.exp(-1j * xi**2 * (L / Zn))

        uout[ii, :] = np.sqrt(-np.log(1 - np.abs(bout)**2)) * np.exp(1j * np.angle(bout))
        m_subplot(Nbursts, xi, uin, ii, uout)

    plt.show()

    # Demodulation and Data Processing (OFDM)
    dataModOut = np.zeros((Nbursts, Nsc), dtype=complex)

    for ii in range(Nbursts):
        c = ifft(fft(psi_xi) * fft(uout[ii, :])) / (mu * Nos)
        c = c[:Nsc]
        dataModOut[ii, :] = np.concatenate([c[:Nsc // 2], c[Nsc // 2 + 2 * Nghost:]])

    # Constellation diagram
    m_plot(dataMod, dataModOut)


def m_plot(dataMod, dataModOut):
    plt.figure()
    plt.scatter(dataMod.real.flatten(), dataMod.imag.flatten(), marker='*', color='white')
    plt.hold(True)
    plt.scatter(dataModOut.real.flatten(), dataModOut.imag.flatten(), marker='o', color='green')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Constellation diagram')
    plt.legend(['Initial optical field', 'Final optical field'], loc='lower right')
    plt.show()




def rrc_impulse(t, Ts, bet):
    temp = bet * np.cos((1 + bet) * np.pi * t / Ts)
    temp += (np.pi * (1 - bet) / 4) * np.sinc((1 - bet) * t / Ts)
    temp /= (1 - (4 * bet * t / Ts)**2)
    Psi = temp * 4 / (np.pi * np.sqrt(Ts))
    poles = np.isinf(temp)
    Psi[poles] = (bet / np.sqrt(Ts)) * (-(2 / np.pi) * np.cos(np.pi *
                                                              (1 + bet) / (4 * bet)) + np.sin(np.pi * (1 + bet) / (4 * bet)))
    return Psi


def m_subplot(Nbursts, xi, uin, ii, uout):
    plt.subplot(1, Nbursts, ii + 1)
    plt.plot(xi, np.abs(uin[ii, :]), xi, np.abs(uout[ii, :]))
    plt.xlabel('Nonlinear frequency, xi')
    plt.ylabel('|u(xi)|')
    plt.legend(['Initial spectrum', 'Final spectrum'], loc='lower right')
    plt.title(f'Burst #{ii+1}')
