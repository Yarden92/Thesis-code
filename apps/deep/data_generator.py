import numpy as np
from tqdm import tqdm

from src.deep import data_loaders
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier


def main():
    # config
    data_len = 10  # for each mu
    mu_len = 3
    num_symbols = 512
    qam = 1024
    dir = f'../../data/qam{qam}_{data_len}x{mu_len}'
    # mu = 1e-3
    # mu_vec = [1e-3, 1e-2, 1e-1, 0.5, 0.9]
    mu_vec = np.linspace(start=0.0005, stop=0.07, num=mu_len)
    cs = ChannelSimulator(m_qam=qam,
                          num_symbols=num_symbols,
                          normalization_factor=0,  # will be overwritten during runtime
                          dt=1,
                          ssf=SplitStepFourier(
                              b2=-20e-27,
                              gamma=0.003,
                              t0=125e-12,
                              dt=1,
                              z_n=1000e3,
                              h=200
                          ),
                          verbose=False)

    # generate the date
    data_loaders.gen_data(data_len, num_symbols, mu_vec, cs, dir, tqdm=tqdm,logger_path='../../logs')


if __name__ == '__main__':
    main()
