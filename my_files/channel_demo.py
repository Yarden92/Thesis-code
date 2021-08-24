import numpy as np

from my_files.src import NFT
from my_files.src import data
from my_files.src import params as p
from my_files.src import pulse_shaping
from my_files.src import signal_processing as sp
from my_files.src import visualizer


def main():
    msg = data.generate_random_msg(p.length_of_msg)  # binary vec [0,1,0,...]
    visualizer.print_bits(msg, p.bps)

    output_msg = simulate_comm_system(msg)
    visualizer.print_bits(output_msg, p.bps)


def simulate_comm_system(msg_bits: np.ndarray) -> np.ndarray:
    """
    msg     ->  [encoder]   ->  [Modulation?]   ->  [pre-equalizer] ->  [pulse-shaping] ->  [INFT]  ⤵
                                                                                                  [channel]
    out_msg <-  [decoder]   <-  [Demodulation?] <-  [equalizer]     <-  [             ] <-  [NFT]   /


    :param msg_bits: input message - binary vector
    :param p: parameters object
    :return: output message after channel - binary vector
    """

    modulated_data = data.data_modulation(msg_bits, p)  # r[xi,0] = [1+j,-1+j,1-j,...]
    visualizer.plot_constellation_map_with_points(modulated_data, p.m_qam)

    normalized_modulated_data = sp.normalize_vec(modulated_data, p.normalization_factor)

    signal = pulse_shaping.pulse_shaping(normalized_modulated_data)
    visualizer.my_plot(np.real(signal), name='signal=X(xi) * h(xi)', ylabel='Re{signal}', xlabel='xi')

    Tmax = p.Tmax  # sp.estimate_T(signal)
    # tx_samples = obsoleted_doing_INFT_in_chuncks(Tmax, parameters, signal)
    tx_signal = NFT.INFT(signal, Tmax)  # q[t,0]
    visualizer.my_plot(tx_signal, name='signal in time', xlabel='t')

    rx_samples = sp.pass_through_channel(tx_signal, p.channel_func)  # q[t,L]
    rx_data = NFT.NFT(rx_samples, BW=2, Tmax=Tmax)  # r[xi,L]
    visualizer.my_plot(np.real(rx_data), name='signal after NFT again', ylabel='Re{signal}', xlabel='xi')
    rx_data_eq = sp.channel_equalizer(rx_data)

    unnormalized_rx_vec = sp.unnormalize_vec(rx_data_eq, p.normalization_factor)
    visualizer.plot_constellation_map_with_points(unnormalized_rx_vec, p.m_qam)

    output_msg = data.data_decoder(unnormalized_rx_vec)

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
    main()
