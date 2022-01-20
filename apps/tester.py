import warnings

import numpy as np

from src import multi_run
from src.channel_simulation import ChannelSimulator
from src.split_step_fourier import SplitStepFourier
from src.visualizer import Visualizer


def test_channel_debug():
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


def test_multi_channel_iterations():
    cs = ChannelSimulator(m_qam=16, verbose=False, channel_func=SplitStepFourier())

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



def test_split():
    cs = ChannelSimulator(m_qam=16,
                          num_symbols=64,
                          normalization_factor=0.001,
                          channel_func=SplitStepFourier(alpha=0),
                          verbose=False)
    ber, num_errors = cs.iterate_through_channel()
    print(f'ber = {ber}, N_errors = {num_errors}')


if __name__ == '__main__':
    test_multi_channel_iterations()
