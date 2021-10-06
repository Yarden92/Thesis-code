import time

import numpy as np

import my_files.src.visualizer
from my_files.src import NFT
from my_files.src import data
from my_files.src import params as p
from my_files.src import pulse_shaping
from my_files.src import signal_processing as sp
from my_files.src import visualizer


def simulate_comm_system() -> None:
    """
    msg ->  [encoder?]  ->  [Modulation + oversampling]   ->  [pulse-shaping] ->  [pre-equalizer] ->  [upsampling] ->  [INFT]  ⤵
                                                                                                            [channel]
    msg <-  [decoder?]  <-  [Demodulation?] <-  [match-filter] <-  [equalizer]     <- [downsamp]   <-  [NFT]   /
    :param msg_bits: input message - binary vector
    :param p: parameters object
    :return: output message after channel - binary vector
    """

    # TODO: multiply the match filter by the filter of rrc

    q = pulse_shaping.example_for_rrc_from_the_internet()
    for i in np.linspace(1, 100000, 100):
        if (NFT.INFT(q, int(i))):
            print(f'number {i} worked!')


    x1i = _gen_msg_()

    x2i = _modulation_(x1i)
    x3i = _upsampling_(x2i)  # zero padding in between
    x4i = _pulse_shaping_(x3i)

    # TODO: 3: padd x3i with zeros before and after
    x5i = _pre_equalizer_(x4i)  # normalize + BPF + zero padding
    x6i = _INFT_(x5i)

    x6o = _channel_(x6i)  # nothing
    x5o = _NFT_(x6o)
    x4o = _downsampling_(x5o)  # (nothing) TODO
    x3o = _equalizer_(x4o)  # de-normalize
    x2o = _de_pulse_shaping_(x3o)  # (nothing) TODO <- what is it? sampling?
    x1o = _demodulation_(x2o)


def _gen_msg_():
    msg = data.generate_random_msg(p.length_of_msg)  # binary vec [0,1,0,...]
    if p.plot_vec_after_creation:
        visualizer.print_bits(msg, p.sps, 'message before channel')
    return msg


def _modulation_(msg_bits):
    modulated_data = data.data_modulation(msg_bits, p.m_qam)  # r[xi,0] = [1+j,-1+j,1-j,...]
    if p.plot_vec_after_creation:
        visualizer.plot_constellation_map_with_points(modulated_data, p.m_qam, 'clean before channel')
    return modulated_data


def _pre_equalizer_(x: np.ndarray) -> np.ndarray:
    """
    normalizing + bpf + zero padding
    :param x: signal vector
    :return: processed signal vector
    """
    xi_axis = NFT.create_xivec(p.Tmax, N_xi=len(x))
    x = sp.normalize_vec(x, p.normalization_factor)
    # x = sp.bpf(x,xi_axis, fstart=p.Filter_Fstart, fstop=p.Filter_Fstop)
    # x = sp.pad_vec(x, before=15e6, after=15e6, padding_value=0)
    if p.plot_vec_after_creation:
        visualizer.my_plot(xi_axis, np.real(x),
                           legend=['Real{X}'],
                           # xi_vec,np.imag(signal),
                           # legend=['Real{X}','Imag{X}'],
                           name='signal after pre equalizer', xlabel='xi')

    return x


def _pulse_shaping_(x):
    y = pulse_shaping.pulse_shaping(x, p.Ts, p.over_sampling, len(x), p.roll_off)
    if p.plot_vec_after_creation:
        xi_vec = NFT.create_xivec(p.Tmax, N_xi=len(y))
        visualizer.my_plot(xi_vec, np.real(y),
                           legend=['Real{X}'],
                           # xi_vec,np.imag(signal),
                           # legend=['Real{X}','Imag{X}'],
                           name='signal after pulse shaping=X(xi) * h(xi)', xlabel='xi')
    return y


def _upsampling_(x):
    y = sp.up_sample(x,factor=p.over_sampling)
    return y


def _INFT_(signal):
    # tx_samples = obsoleted_doing_INFT_in_chuncks(Tmax, parameters, signal)
    tx_signal = NFT.INFT(signal, p.Tmax)  # q[t,0]
    if p.plot_vec_after_creation:
        tvec = NFT.create_tvec(p.Tmax, N_time=len(tx_signal))
        visualizer.my_plot(tvec, np.abs(tx_signal), name=f'signal in time, (T={p.Tmax})', xlabel='t', legend='|x(t)|')
        # visualizer.my_plot(np.abs(NFT.IFFT(signal)), name='normal ifft for comparison', xlabel='t', legend='|x(t)|')
    return tx_signal


def _channel_(tx_signal):
    rx_samples = sp.pass_through_channel(tx_signal, p.channel_func)  # q[t,L]
    return rx_samples


def _NFT_(rx_samples):
    # for bw in [2,6,10,20,40,60,100,300,1000,20e3,600e3,3e6]:
    rx_data = NFT.NFT(rx_samples, BW=p.BW, Tmax=p.Tmax)  # r[xi,L]
    if p.plot_vec_after_creation:
        visualizer.my_plot(np.real(rx_data), name=f'signal after NFT again, (BW={p.BW})', ylabel='Re{signal}', xlabel='xi')
    return rx_data


def _downsampling_(x):
    return x


def _equalizer_(x):
    x = sp.unnormalize_vec(x, p.normalization_factor)
    x = sp.channel_equalizer(x)
    return x


def _de_pulse_shaping_(x):
    # y = pulse_shaping.pulse_shaping(x)
    if p.plot_vec_after_creation:
        my_files.src.visualizer.eye_diagram(x, sps=p.sps)
        # visualizer.plot_constellation_map_with_points(y, p.m_qam, 'after depulse shaping')
    return x


def _demodulation_(unnormalized_rx_vec):
    output_msg = data.data_demodulation(unnormalized_rx_vec, p.m_qam)
    if p.plot_vec_after_creation:
        visualizer.print_bits(output_msg, p.sps, 'message after channel')
    return output_msg


def obsoleted_doing_INFT_in_chuncks(Tmax, parameters, signal):
    folded_signal = sp.fold_subvectors(signal, N=parameters.NFT_size)
    tx_samples = []
    for sub_signal in folded_signal:
        visualizer.my_plot(np.real(sub_signal), name='sub_signal', ylabel='Re{sub_signal}')
        tx_subsignal = NFT.INFT(sub_signal, Tmax)  # q[t,0]
        tx_samples.append(tx_subsignal)
    return tx_samples


if __name__ == "__main__":
    simulate_comm_system()
