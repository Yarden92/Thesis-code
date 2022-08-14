import os
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.general_methods.visualizer import Visualizer
from src.optics.channel_simulation import ChannelSimulator
from src.optics.split_step_fourier import SplitStepFourier

x_file_name = 'data_x.npy'
y_file_name = 'data_y.npy'
conf_file_name = 'conf.json'


def read_folder(dir, verbose=True):
    # example: dir = f'data/10_samples_mu=0.001'

    x_path = f'{dir}/{x_file_name}'
    y_path = f'{dir}/{y_file_name}'
    conf_path = f'{dir}/{conf_file_name}'
    # read
    with open(x_path, 'rb') as f:
        all_x_read = np.load(f)
    with open(y_path, 'rb') as f:
        all_y_read = np.load(f)
    with open(conf_path, 'r') as f:
        conf_read = json.load(f)

    if verbose:
        print('data loaded.')

    return all_x_read, all_y_read, conf_read


def save_data(dir, all_x, all_y, cs, verbose=True):
    x_path = f'{dir}/{x_file_name}'
    y_path = f'{dir}/{y_file_name}'
    conf_path = f'{dir}/{conf_file_name}'
    # save to files
    Path(dir).mkdir(parents=True, exist_ok=True)

    with open(x_path, 'wb') as f:
        np.save(f, all_x)
        if verbose: print(f'saved inputs to: {x_path}')
    with open(y_path, 'wb') as f:
        np.save(f, all_y)
        if verbose: print(f'saved outputs to: {y_path}')
    with open(conf_path, 'w') as f:
        json.dump(cs.params_to_dict(), f, indent=4)
        # print(f'saved json params to: {conf_path}')


def calc_ber_for_folder(all_x_read, all_y_read, conf_read, verbose=True):
    num_errors = 0
    ber_vec = []
    cs = ChannelSimulator.from_dict(conf_read)
    for i, (x, y) in enumerate(zip(all_x_read, all_y_read)):
        msg_in = cs.steps8_to_10(y)
        msg_out = cs.steps8_to_10(x)
        ber_i, num_errors_i = cs.cb.calc_ber(msg_in, msg_out, cs.length_of_msg, cs.sps)
        ber_vec.append(ber_i)
        num_errors += num_errors_i
        if verbose:
            print(f'data {i} has ber of: {ber_i} with {num_errors_i}/{cs.length_of_msg} bit errors')
    return ber_vec, num_errors


def gen_data(data_len, num_symbols, mu_vec, cs, root_dir='data', tqdm=tqdm):
    vec_lens = num_symbols*cs.over_sampling + cs.N_rrc - 1
    assert vec_lens == num_symbols*8*2, "the formula is not correct! check again"
    pbar = tqdm(total=len(mu_vec)*data_len)
    for mu in mu_vec:
        dir = f'{root_dir}/{data_len}_samples_mu={mu:.3f}'
        # all_x = np.zeros(shape=(data_len, vec_lens),dtype=complex)
        # all_y = np.zeros(shape=(data_len, vec_lens),dtype=complex)
        all_x = []
        all_y = []
        cs.normalization_factor = mu
        for i in range(data_len):
            x, y = cs.gen_io_data()
            # all_x[i], all_y[i] = x, y
            all_x.append(x)
            all_y.append(y)
            pbar.update()
        save_data(dir, all_x, all_y, cs, False)


def gen_ber_mu_from_folders(root_dir, sub_name, verbose_level=0):
    # walk through folder [data] and search for folders that named as [10_samples_****]
    mu_vec, ber_vec = [], []
    for dirpath, _, _ in tqdm(os.walk(root_dir)):
        if sub_name in dirpath:
            mu = float(dirpath.split(sub_name)[-1])
            all_x_read, all_y_read, conf_read = read_folder(dirpath, verbose_level >= 1)
            sub_ber_vec, num_errors = calc_ber_for_folder(all_x_read, all_y_read, conf_read, verbose_level >= 2)
            ber = np.sum(sub_ber_vec)/len(sub_ber_vec)

            mu_vec.append(mu)
            ber_vec.append(ber)

    indices = np.argsort(mu_vec)
    mu_vec = np.array(mu_vec)[indices]
    ber_vec = np.array(ber_vec)[indices]

    return ber_vec, mu_vec


def test1():
    # test ber of folder
    print(os.getcwd())
    dir = f'../../apps/deep/data/3_samples_mu=0.9'
    all_x_read, all_y_read, conf_read = read_folder(dir, False)
    ber_vec, num_errors = calc_ber_for_folder(all_x_read, all_y_read, conf_read)
    print(f'the avg ber is {np.mean(ber_vec)}')


def test2():
    # test generated data
    # config
    data_len = 3  # for each mu
    num_symbols = 512
    # mu = 1e-3
    mu_vec = [1e-3, 1e-2, 1e-1, 0.5, 0.9]
    cs = ChannelSimulator(m_qam=16,
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
    gen_data(data_len, num_symbols, mu_vec, cs)


def test3():
    # test ber vs mu from folders
    sub_name = '3_samples_mu='
    ber_vec, mu_vec = gen_ber_mu_from_folders('../../apps/deep/data/qam1024', sub_name)
    indices = np.argsort(mu_vec)
    Visualizer.plot_bers(np.array(mu_vec)[indices], [np.array(ber_vec)[indices]], [sub_name])


if __name__ == '__main__':
    test3()
