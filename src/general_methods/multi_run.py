import numpy as np
from tqdm import tqdm

from src.optics.channel_simulation import ChannelSimulator
from src.optics.channel_simulation2 import ChannelSimulator2


def create_us_vec(n_steps=10, min_u=-2.5, max_u=-1):
    normalizing_factors = 10 ** np.linspace(min_u, max_u, n_steps)
    return normalizing_factors


def run_n_times(cs: ChannelSimulator, n=10, pbar=None) -> (float, int):
    # outputs BER from N realisations
    num_errors = 0
    for r in range(n):
        try:
            num_errors += cs.iterate_through_channel()[1]
        except Exception as e:
            print(f'Error in run {r}: {e}')
            num_errors += cs.length_of_msg
        if pbar: pbar.update(1)
    ber = num_errors / (n * cs.length_of_msg)
    return ber, num_errors

def run_n_times2(cs: ChannelSimulator2, n=10, pbar=None) -> (float, int):
    # outputs BER from N realisations
    num_errors = 0
    for r in range(n):
        try:
            num_errors += cs.simulate_and_analyze()[1]
        except Exception as e:
            print(f'Error in run {r}: {e}')
            num_errors += cs.block11.N_bin
        if pbar: pbar.update(1)
    ber = num_errors / (n * cs.block11.N_bin)
    return ber, num_errors


def ber_vs_us(cs, normalizing_factors, n_realisations=10,
              pbar_title=None):
    bers = []
    num_errors_vec = []
    pbar = tqdm(total=n_realisations * len(normalizing_factors), leave=True, position=0)
    if pbar_title: pbar.set_description_str(pbar_title)
    for i, u in enumerate(normalizing_factors):
        cs.normalization_factor = u
        ber_i, num_errors_i = run_n_times(cs, n_realisations, pbar)
        bers.append(ber_i)
        num_errors_vec.append(num_errors_i)
    errs = np.array(num_errors_vec)
    return np.array(bers), errs
