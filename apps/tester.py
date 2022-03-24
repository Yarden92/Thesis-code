import warnings

import numpy as np

from src.general_methods import multi_run
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier
from src.general_methods.visualizer import Visualizer


def test1_channel_debug():
    cs = ChannelSimulator(m_qam=16,
                          num_symbols=64,
                          normalization_factor=1,
                          verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ber, num_errors = cs.iterate_through_channel()
        x = np.abs(cs.x[5])
        print(f'mean = {x.mean():.2f} , min = {x.min():.2f} , max = {x.max():.2f} ')

    print(f'ber = {ber}, N_errors = {num_errors}')


def test2_multi_channel_iterations():
    cs = ChannelSimulator(m_qam=16, verbose=False, ssf=SplitStepFourier())

    num_realisations = 2
    us_vec = multi_run.create_us_vec(n_steps=10, min_u=-6, max_u=-1)
    N_symbols_vec = np.array([64, 128, 256, 512])

    with np.printoptions(precision=2):
        print(f'normalizing factors are: {us_vec}')

        bers_vec, legends = [], []
        for n_sym in N_symbols_vec:
            cs.num_symbols = n_sym
            bers, errs = multi_run.ber_vs_us(cs, us_vec, n_realisations=num_realisations)
            print(f'{n_sym} symbols: errors found = {errs} / [{cs.length_of_msg * num_realisations} '
                  f'= {cs.length_of_msg} bits * {num_realisations} realisations]')
            bers_vec.append(bers)
            legends.append(f'{n_sym} symbols')

    Visualizer.plot_bers(us_vec, bers_vec, legends)



def test3_split():
    cs = ChannelSimulator(m_qam=16,
                          num_symbols=64,
                          normalization_factor=0.001,
                          ssf=SplitStepFourier(alpha=0),
                          verbose=False)
    ber, num_errors = cs.iterate_through_channel()
    print(f'ber = {ber}, N_errors = {num_errors}')

def test4_import():
    help('modules ModulationPy')

    print('success')

def test5_full_run():
    cs = ChannelSimulator(verbose=False)
    cs.iterate_through_channel()

def test_os():
    x = np.array([1,2,3,4])
    os = 3
    y = over_sample(x,os)
    print(y)

def over_sample(x, over_sampling):  # step 2
    y = np.zeros(over_sampling * len(x), dtype=np.complex64)
    y[::over_sampling] = x

    Visualizer.my_plot(np.real(y[0:50]), name='zero padded - i (real)', function='stem')
    print(f'vec length = {len(y)}, over_sampling period = {over_sampling}')

    return y

if __name__ == '__main__':
    test5_full_run()

