from typing import List

import numpy as np
from ModulationPy import ModulationPy
from matplotlib import pyplot as plt


# from lib.packages.eyediagram.eyediagram.mpl import eyediagram


# from my_files.src.params import Params

#
class Visualizer:
    @staticmethod
    def plot_constellation_map_with_points(data_vec, m_qam, title='after channel'):
        modem = ModulationPy.QAMModem(m_qam)
        fig = Visualizer.plot_constellation_map_grid(modem)

        i, q = np.real(data_vec), np.imag(data_vec)
        plt.plot(i, q, '.')
        plt.xlabel('real part')
        plt.ylabel('imag part')
        plt.title(f'{m_qam}-QAM constellation map {title}')
        plt.show()
        # file_name = 'output/constellation_through_channel.jpg'
        # plt.savefig(file_name)
        # print(f'saved figure under: {file_name}')

    @staticmethod
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

    @staticmethod
    def gen_fig() -> plt.Figure:
        return plt.figure()

    @staticmethod
    def my_plot(*args, name='graph', title=None, output_name=None,
                xlabel=None, ylabel=None,
                fig=None, ax=None, is_show=True,
                legend=None):
        # matplotlib.use('Agg')
        fig = fig or plt.figure()
        ax = ax or plt
        ax.plot(*args)
        if title:
            fig.suptitle(title)
        elif name:
            fig.suptitle(name)

        if xlabel:
            ax.xlabel(xlabel)
        if ylabel:
            ax.ylabel(ylabel)
        if legend:
            ax.legend(legend)

        ax.grid(True)

        # output_name = output_name or name
        # path = os.path.join(p.visualization_path, output_name)
        # plt.savefig(path)
        if is_show:
            plt.show()

    # def multi_plot(x_axes: List[np.ndarray], y_axes: List[np.ndarray], names: List[str]):
    @staticmethod
    def multi_plot(dict_list: List[dict]):
        N = len(dict_list)

        fig, axes = plt.subplots(nrows=N, ncols=1)
        for i, inputs in enumerate(dict_list):
            args = inputs.get('args')
            name = inputs.get('name')
            title = inputs.get('title')
            output_name = inputs.get('output_name')
            xlabel = inputs.get('xlabel')
            ylabel = inputs.get('ylabel')
            legend = inputs.get('legend')
            Visualizer.my_plot(args,
                    name=name, title=title, output_name=output_name,
                    xlabel=xlabel, ylabel=ylabel, legend=legend,
                    fig=fig, ax=axes[i], is_show=False)

        plt.show()

        # assert len(x_axes) == N, "number of x_axes must be equal to number of y_axes"
        #
        # for ax, x_axis, y_axis, names in zip(axes, x_axes, y_axes, names):
        #     ax.plot()

    @staticmethod
    def plot_bins(bins, name='graph'):
        fig = plt.figure()
        plt.bar(bins)
        plt.grid(True)
        plt.show()

    @staticmethod
    def print_bits(bits, M, title='the bits are:'):
        print('\n_______________________________________________')
        print(title, f'- len={len(bits)}')
        mat = np.reshape(bits, (-1, M))
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
        # eyediagram(x,2*sps)
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
