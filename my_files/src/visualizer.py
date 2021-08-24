import os

import numpy as np
from ModulationPy import ModulationPy
from matplotlib import pyplot as plt

from my_files.src import params as p


# from my_files.src.params import Params

#
def plot_constellation_map_with_points(data_vec, params):
    modem = ModulationPy.QAMModem(params.m_qam)
    fig = plot_constellation_map_grid(modem)

    i, q = np.real(data_vec), np.imag(data_vec)
    plt.plot(i, q, '.')
    plt.xlabel('real part')
    plt.ylabel('imag part')
    plt.title(f'{params.m_qam}-QAM constellation map after channel')
    file_name = 'output/constellation_through_channel.jpg'
    # plt.savefig(file_name)
    plt.show()
    # print(f'saved figure under: {file_name}')


def plot_constellation_map_grid(modem: ModulationPy):
    if modem.M <= 16:
        limits = np.log2(modem.M)
        size = 'small'
    elif modem.M == 64:
        limits = 1.5 * np.log2(modem.M)
        size = 'x-small'
    else:
        limits = 2.25 * np.log2(modem.M)
        size = 'xx-small'

    const = modem.code_book
    fig = plt.figure(figsize=(6, 4), dpi=150)
    for i in list(const):
        x = np.real(const[i])
        y = np.imag(const[i])
        plt.plot(x, y, 'o', color='red')
        if x < 0:
            h = 'right'
            xadd = -.05
        else:
            h = 'left'
            xadd = .05
        if y < 0:
            v = 'top'
            yadd = -.05
        else:
            v = 'bottom'
            yadd = .05
        if (abs(x) < 1e-9 and abs(y) > 1e-9):
            h = 'center'
        elif abs(x) > 1e-9 and abs(y) < 1e-9:
            v = 'center'
        plt.annotate(i, (x + xadd, y + yadd), ha=h, va=v, size=size)
    M = str(modem.M)
    if modem.gray_map == True:
        mapping = 'Gray'
    else:
        mapping = 'Binary'

    if modem.bin_input == True:
        inputs = 'Binary'
    else:
        inputs = 'Decimal'

    plt.grid()
    plt.axvline(linewidth=1.0, color='black')
    plt.axhline(linewidth=1.0, color='black')
    plt.axis([-limits, limits, -limits, limits])
    plt.title(M + '-QAM, Mapping: ' + mapping + ', Input: ' + inputs)
    return fig


def my_plot(*args, name='graph', title=None, output_name=None,
            xlabel=None, ylabel=None,
            legend=None):
    # matplotlib.use('Agg')
    fig = plt.figure()
    plt.plot(*args)
    if title:
        fig.suptitle(title)
    elif name:
        fig.suptitle(name)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)

    if output_name is None:
        output_name = name

    plt.grid(True)

    path = os.path.join(p.path, output_name)
    # plt.savefig(path)

    plt.show()


def plot_bins(bins, name='graph'):
    fig = plt.figure()
    plt.bar(bins)
    plt.grid(True)
    plt.show()


def print_bits(bits, M):
    mat = np.reshape(bits, (-1, M))
    print(mat)
