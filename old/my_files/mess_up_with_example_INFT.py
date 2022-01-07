import numpy as np
from ModulationPy import ModulationPy

from old.my_files.src import visualizer, params
from old.my_files.src import pulse_shaping, NFT


def messing_with_inft_example():
    # set values
    M = 8000  # xi domain length - default = 2048
    D = 2048  # time domain length - default = 1024
    tvec = np.linspace(-2, 2, D)  # default = -2, 2, D

    # get the frequency intervall suited for the given time vector
    rv, XI = NFT.nsev_inverse_xi_wrapper(D, tvec[0], tvec[-1], M)
    xivec = XI[0] + np.arange(M) * (XI[1] - XI[0]) / (M - 1)

    alpha = 2.0
    beta = 0.55
    gamma = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    # set continuous spectrum
    contspec = alpha / (xivec - beta * 1.0j)

    # set discrete spectrum
    bound_states = []  # default: np.array([1.0j * beta])
    normconst_or_residues = np.array([-1.0j * alpha / (gamma + beta)])

    # print params
    print(f't_vec = [{tvec[0]}:{tvec[-1]}] , len={len(tvec)} and D = {D} (time domain length)')
    print(f'xi_vec = [{xivec[0]:.2f}:{xivec[-1]:.2f}] , len={len(xivec)} and M = {M} (xi domain length)')

    # call function
    res = NFT.nsev_inverse(xivec, tvec, contspec, bound_states, normconst_or_residues)

    if res['return_value'] == 0:
        print(f'q^ after INFT - len={len(res["q"])}')
        print('\n[V] sucess!')
        print()


def example_from_internet_for_pulse_shaping():
    N = 1024  # Number of symbols
    os = 8  # over sampling factor
    # Create modulation. QAM16 makes 4 bits/symbol
    mod1 = ModulationPy.QAMModem(16)
    bps = 4
    # Generate the bit stream for N symbols
    sB = np.random.randint(0, 2, N * bps)
    # Generate N complex-integer valued symbols
    sQ = mod1.modulate(sB)
    sQ_upsampled = np.zeros(os * (len(sQ) - 1) + 1, dtype=np.complex64)
    sQ_upsampled[::os] = sQ  # TODO: there is no need for over sampling, theres already parameter (osf in nsev_inverse)
    # Create a filter with limited bandwidth. Parameters:
    #      N: Filter length in samples
    #    0.8: Roll off factor alpha
    #      1: Symbol period in time-units
    #     24: Sample rate in 1/time-units
    sPSF = pulse_shaping.rrcosfilter(N, alpha=0.8, Ts=1, Fs=os)[1]
    # Analog signal has N/2 leading and trailing near-zero samples
    c = np.convolve(sPSF, sQ_upsampled)

    q_tag = NFT.INFT(c, params.Tmax)
    visualizer.my_plot(q_tag, name=f'q after INFT with Tmax={params.Tmax}')


example_from_internet_for_pulse_shaping()
