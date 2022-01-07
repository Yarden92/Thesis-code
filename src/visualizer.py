import matplotlib.pyplot as plt
import numpy as np
from ModulationPy import ModulationPy
from matplotlib.axes import Axes


class Visualizer:
    @staticmethod
    def plot_constellation_map_grid(modem: ModulationPy):
        size = 'small' if modem.M <= 16 else 'x-small' if modem.M == 64 else 'xx-small'
        logM = np.log2(modem.M)
        limits = logM if modem.M <= 16 else 1.5 * logM if modem.M == 64 else 2.25 * logM

        const = modem.code_book
        fig = plt.figure(figsize=(6, 4), dpi=150)
        for i in list(const):
            x = np.real(const[i])
            y = np.imag(const[i])
            plt.plot(x, y, 'o', color='red')

            xadd, h = (-.05, 'right') if x < 0 else (.05, 'left')
            yadd, v = (-.05, 'top') if y < 0 else (.05, 'bottom')

            if (abs(x) < 1e-9 and abs(y) > 1e-9):
                h = 'center'
            elif abs(x) > 1e-9 and abs(y) < 1e-9:
                v = 'center'
            plt.annotate(i, (x + xadd, y + yadd), ha=h, va=v, size=size)
        M = str(modem.M)

        mapping = 'Gray' if modem.gray_map else 'Binary'
        inputs = 'Binary' if modem.bin_input else 'Decimal'

        plt.grid()
        plt.axvline(linewidth=1.0, color='black')
        plt.axhline(linewidth=1.0, color='black')
        plt.axis([-limits, limits, -limits, limits])
        plt.title(M + '-QAM, Mapping: ' + mapping + ', Input: ' + inputs)
        return fig

    @staticmethod
    def plot_constellation_map_with_points(data_vec, m_qam,
                                           title_ending='after channel'):
        modem = ModulationPy.QAMModem(m_qam)
        fig = Visualizer.plot_constellation_map_grid(modem)

        i, q = np.real(data_vec), np.imag(data_vec)
        plt.plot(i, q, '.')
        plt.xlabel('real part')
        plt.ylabel('imag part')
        plt.title(f'{m_qam}-QAM constellation map {title_ending}')
        plt.show()

    @staticmethod
    def my_plot(*args, name='graph', title=None, output_name=None,
                xlabel=None, ylabel=None, legend=None,
                custom_keyval=None,
                hold=False, function='plot', ax=None):
        ax: Axes = ax or plt.subplot()
        getattr(ax, function)(*args)
        ax.grid(True)
        ax.set_title(title or name)
        if xlabel:  ax.set_xlabel(xlabel)
        if ylabel:  ax.set_ylabel(ylabel)
        if legend:  ax.legend(legend)
        if custom_keyval: getattr(ax, custom_keyval[0])(custom_keyval[1])
        if not hold: plt.show()

    @staticmethod
    def print_bits(bits, M, title='the bits are:'):
        print('\n_______________________________________________')
        print(title, f'- len={len(bits)}')
        mat = np.int8(np.reshape(bits, (-1, M)))
        print(mat)
        print('\n')

    @staticmethod
    def eye_diagram(x: np.ndarray, sps: int) -> None:
        """
        creates an eye_diagram from vector x
        :param x: analog vector - after pulse shaping - received at the receiver
        :param sps: samples per symbol - do determine the window size for cropping
        :return: None (new eye-diagram plot will be generated)
        """
        fig = plt.figure()
        num_plots = int(len(x) / sps)
        for i in range(num_plots):
            index1 = i * sps
            index2 = (i + 1) * sps
            sub_x = x[index1:index2]
            plt.plot(np.real(sub_x))
        plt.title('eye diagram')
        plt.show()

    @staticmethod
    def print_nft_options(res_ob: dict) -> None:
        """
        pretty print the options of the INFT / NFT that was done.
        :param res_ob: the res output of the function nsev_inverse
        :return: None (pretty prints the options)
        """
        assert isinstance(res_ob['options'], str), "non valid object, insert the outcome of nsev_inv"
        jsonable_str = ('{' + res_ob['options'] + '}').replace("\'", '\"')
        import json
        json_ob = json.loads(jsonable_str)
        print(json.dumps(json_ob, indent=4))