import numpy as np

from my_files.src import NFT
from my_files.src import data
from my_files.src import params as p
from my_files.src import pulse_shaping
from my_files.src import signal_processing as sp
from my_files.src import visualizer


def simulate_comm_system() -> None:
    """
    msg ->  [encoder?]  ->  [Modulation?]   ->  [pulse-shaping] ->  [pre-equalizer] ->  [upsampling] ->  [INFT]  ⤵
                                                                                                            [channel]
    msg <-  [decoder?]  <-  [Demodulation?] <-  [             ] <-  [equalizer]     <- [downsamp]   <-  [NFT]   /


    :param msg_bits: input message - binary vector
    :param p: parameters object
    :return: output message after channel - binary vector
    """

    x1i = _gen_msg_()

    x2i = _modulation_(x1i)
    x3i = _pulse_shaping_(x2i)
    x4i = _pre_equalizer_(x3i)  # normalize
    x5i = _upsampling_(x4i)  # (nothing) TODO
    x6i = _INFT_(x5i)

    x6o = _channel_(x6i)  # nothing
    x5o = _NFT_(x6o)
    x4o = _downsampling_(x5o)  # (nothing) TODO
    x3o = _equalizer_(x4o)  # de-normalize
    x2o = _depulse_shaping_(x3o)  # (nothing) TODO
    x1o = _demodulation_(x2o)


def _gen_msg_():
    msg = data.generate_random_msg(p.length_of_msg)  # binary vec [0,1,0,...]
    if p.plot_vec_after_creation:
        visualizer.print_bits(msg, p.bps, 'message before channel')
    return msg


def _modulation_(msg_bits):
    modulated_data = data.data_modulation(msg_bits, p.m_qam)  # r[xi,0] = [1+j,-1+j,1-j,...]
    if p.plot_vec_after_creation:
        visualizer.plot_constellation_map_with_points(modulated_data, p.m_qam, 'clean before channel')
    return modulated_data


def _pre_equalizer_(modulated_data):
    normalized_modulated_data = sp.normalize_vec(modulated_data, p.normalization_factor)
    return normalized_modulated_data


def _pulse_shaping_(normalized_modulated_data):
    signal = pulse_shaping.pulse_shaping(normalized_modulated_data)
    if p.plot_vec_after_creation:
        xi_vec = NFT.create_xivec(p.Tmax,N_xi=len(signal))
        visualizer.my_plot(xi_vec,np.real(signal),
                           xi_vec,np.imag(signal),
                           legend=['Real{X}','Imag{X}'],
                           name='signal=X(xi) * h(xi)', ylabel='Re{signal}', xlabel='xi')
    return signal


def _upsampling_(x):
    return x


def _INFT_(signal):
    # tx_samples = obsoleted_doing_INFT_in_chuncks(Tmax, parameters, signal)
    tx_signal = NFT.INFT(signal, p.Tmax)  # q[t,0]
    if p.plot_vec_after_creation:
        visualizer.my_plot(tx_signal, name='signal in time', xlabel='t')
    return tx_signal


def _channel_(tx_signal):
    rx_samples = sp.pass_through_channel(tx_signal, p.channel_func)  # q[t,L]
    return rx_samples


def _NFT_(rx_samples):
    rx_data = NFT.NFT(rx_samples, BW=2, Tmax=p.Tmax)  # r[xi,L]
    if p.plot_vec_after_creation:
        visualizer.my_plot(np.real(rx_data), name='signal after NFT again', ylabel='Re{signal}', xlabel='xi')
    return rx_data


def _downsampling_(x):
    return x


def _equalizer_(x):
    x = sp.unnormalize_vec(x, p.normalization_factor)
    x = sp.channel_equalizer(x)
    return x


def _depulse_shaping_(x):
    if p.plot_vec_after_creation:
        visualizer.plot_constellation_map_with_points(x, p.m_qam, 'after channel')
    return x


def _demodulation_(unnormalized_rx_vec):
    output_msg = data.data_demodulation(unnormalized_rx_vec, p.m_qam)
    if p.plot_vec_after_creation:
        visualizer.print_bits(output_msg, p.bps, 'message after channel')
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
