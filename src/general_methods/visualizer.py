import matplotlib.pyplot as plt
import numpy as np
from ModulationPy import ModulationPy
from matplotlib.axes import Axes

from src.general_methods.signal_processing import SP


class Visualizer:
    @staticmethod
    def plot_constellation_map_grid(modem: ModulationPy):
        size = 'small' if modem.M <= 16 else 'x-small' if modem.M == 64 else 'xx-small'
        logM = np.log2(modem.M)
        limits = logM if modem.M <= 16 else 1.5*logM if modem.M == 64 else 2.25*logM

        const = modem.code_book
        fig = plt.figure(figsize=(6, 4), dpi=100)
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
    def plot_constellation_map_with_3_data_vecs(data_vec, data_vec2, data_vec3, m_qam,
                                                title_ending, legend):
        modem = ModulationPy.QAMModem(m_qam)
        fig = Visualizer.plot_constellation_map_grid(modem)

        i, q = np.real(data_vec), np.imag(data_vec)
        plt.plot(i, q, '.', label=legend[0])
        i, q = np.real(data_vec2), np.imag(data_vec2)
        plt.plot(i, q, '.', label=legend[1])
        i, q = np.real(data_vec3), np.imag(data_vec3)
        plt.plot(i, q, '.', label=legend[2])
        plt.xlabel('real part')
        plt.ylabel('imag part')
        plt.title(f'{m_qam}-QAM constellation map {title_ending}')
        plt.legend()
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
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend(legend)
        if custom_keyval:
            getattr(ax, custom_keyval[0])(custom_keyval[1])
        if output_name:
            plt.savefig(output_name)
        if not hold and not output_name:
            plt.show()

    @staticmethod
    def double_plot(title: str, y1, y2, x1_vec=None, x2_vec=None,
                    name1: str = 'plot1', name2: str = 'plot2', function='plot', output_name=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        if title:
            fig.suptitle(title)
        if x1_vec is None:
            x1_vec = np.arange(len(y1))
        if x2_vec is None:
            x2_vec = np.arange(len(y2))
        Visualizer.my_plot(x1_vec, y1, name=name1, ax=ax1, function=function, hold=True)
        Visualizer.my_plot(x2_vec, y2, name=name2, ax=ax2,
                           function=function, output_name=output_name)

    @staticmethod
    def twin_zoom_plot(title: str, full_y, zoom_indices, x_vec=None, xlabel='index', function='plot', output_name=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        if title:
            fig.suptitle(title)
        if x_vec is None:
            x_vec = np.arange(len(full_y))
        Visualizer.my_plot(x_vec, full_y, name=f'full scale', xlabel=xlabel, ax=ax1, function=function, hold=True)
        Visualizer.my_plot(x_vec[zoom_indices], full_y[zoom_indices], name=f'crop in', xlabel=xlabel, ax=ax2,
                           function=function, output_name=output_name)

    @staticmethod
    def twin_zoom_plot_vec(title: str, y_vecs, legends, zm_indx, x_vec=None, xlabel='index', function='plot'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        if title:
            fig.suptitle(title)
        if x_vec is None:
            x_vec = np.arange(len(y_vecs))
        Visualizer.my_plot(x_vec, y_vecs, name='full scale', xlabel=xlabel, ax=ax1, function=function, hold=True,
                           legend=legends)
        Visualizer.my_plot(x_vec[zm_indx], y_vecs[:, zm_indx], name='crop in', xlabel=xlabel, ax=ax1, function=function,
                           legend=legends)

        for i, (y, l) in enumerate(zip(y_vecs, legends)):
            hold = i != len(y_vecs) - 1
            Visualizer.my_plot(x_vec, y, name=f'full scale', xlabel=xlabel, ax=ax1, legend=l, function=function,
                               hold=True)
            Visualizer.my_plot(x_vec[zm_indx], y[zm_indx], name=f'crop in', xlabel=xlabel, ax=ax2, legend=l,
                               function=function, hold=hold)

    @staticmethod
    def data_trio_plot(y1, y2, y3, zoom_indices=None, title: str = None, x_vec=None, xlabel='index',
                       function='plot', 
                       names=['input (dirty)', 'output (clean)', 'pred']):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
        if title:
            fig.suptitle(title)
        if x_vec is None:
            x_vec = np.arange(len(y1))
        if zoom_indices is None:
            zoom_indices = range(len(x_vec))

        x = x_vec[zoom_indices]
        y1 = y1[zoom_indices]
        y2 = y2[zoom_indices]
        y3 = y3[zoom_indices]

        Visualizer.my_plot(x, y1, name=names[0], xlabel=xlabel, ax=ax1, function=function, legend=['real', 'imag'],
                           hold=True)
        Visualizer.my_plot(x, y2, name=names[1], xlabel=xlabel, ax=ax2, function=function, legend=['real', 'imag'],
                           hold=True)
        Visualizer.my_plot(x, y3, name=names[2], xlabel=xlabel, ax=ax3, function=function, legend=['real', 'imag'])

    @staticmethod
    def print_bits(bits, M, title='the bits are:'):
        print('\n_______________________________________________')
        print(title, f'- len={len(bits)}')
        # mat = np.int8(np.reshape(bits, (-1, M)))
        mat = np.reshape(bits, (-1, M))
        print(mat)
        # print('\n')

    @staticmethod
    def eye_diagram(x: np.ndarray, sps: int) -> None:
        """
        creates an eye_diagram from vector x
        :param x: analog vector - after pulse shaping - received at the receiver
        :param sps: samples per symbol - do determine the window size for cropping
        :return: None (new eye-diagram plot will be generated)
        """
        fig = plt.figure()
        num_plots = int(len(x)/sps)
        for i in range(num_plots):
            index1 = i*sps
            index2 = (i + 1)*sps
            sub_x = x[index1:index2]
            plt.plot(np.real(sub_x))
        plt.title('eye diagram')
        plt.show()

    @staticmethod
    def print_signal_specs(x: np.ndarray, t_vec: np.ndarray, th=None) -> None:
        power = SP.signal_power(x)
        print(f'signal power = {power:.2e}')

        th = th or SP.peak(x)*0.01
        tmin = t_vec[np.min(np.where(np.abs(x) > th))]
        tmax = t_vec[np.max(np.where(np.abs(x) > th))]

        print(f'signal bw = [{tmin:.2e}:{tmax:.2e}]')

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

    @staticmethod
    def plot_bers(us, bers_vecs, legends=None, output_path=None):
        plt.figure(figsize=[10, 5])
        for bers in bers_vecs:
            mean = bers.mean(axis=-1)
            std = bers.std(axis=-1)
            plt.semilogy(us, bers)
            # plt.fill_between(us,mean-std,mean+std,alpha=0.4)

        plt.xlabel('normalizing factor'), plt.ylabel('BER')
        plt.title('BER vs normalizing factor')
        plt.grid(which='both', axis='y')
        plt.grid(which='major', axis='x')
        # plt.ylim(top=1,bottom=3e-4)
        if legends:
            plt.legend(legends)
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @staticmethod
    def plot_bers_boxplot(us, bers_vecs, legends=None, output_path=None):
        plt.figure(figsize=[15, 5])
        plt.boxplot(bers_vecs, labels=us, patch_artist=True,
                    boxprops=dict(facecolor='pink', color='black'),
                    medianprops=dict(color='black'),
                    )
        plt.yscale('log')
        plt.xlabel('normalizing factor'), plt.ylabel('BER')
        plt.xticks(rotation=45)
        plt.title(f'BER vs normalizing factor with {len(bers_vecs)} permutations each')
        plt.grid(which='both', axis='y')
        plt.grid(which='major', axis='x')
        if legends:
            plt.legend(legends)
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @staticmethod
    def plot_histogram_of_bers(bers_vecs, legends=None, output_path=None):
        plt.figure(figsize=[15, 5])
        plt.hist(bers_vecs, bins=100, label=legends)
        plt.xlabel('BER'), plt.ylabel('count')
        plt.title(f'BER histogram with {len(bers_vecs)} permutations each')
        plt.grid(which='both', axis='y')
        plt.grid(which='major', axis='x')
        if legends:
            plt.legend(legends)
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @staticmethod
    def plot_loss_vec(train_loss_vec, valid_loss_vec):
        assert len(train_loss_vec) == len(valid_loss_vec), "train and valid loss vectors must have the same length"
        x = range(len(train_loss_vec))
        plt.plot(x, train_loss_vec, label='train')
        plt.plot(x, valid_loss_vec, label='valid')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True)
        plt.show()
