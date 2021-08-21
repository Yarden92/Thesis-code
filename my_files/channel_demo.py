import numpy as np

from my_files.src import NFT
from my_files.src import pulse_shaping
from my_files.src import visualizer
from my_files.src import params as p
from my_files.src import data
from my_files.src import signal_processing as sp


def main():
    msg = data.generate_random_msg(p.length_of_msg)  # binary vec [0,1,0,...]
    # visualizer.my_plot(msg[0:20],'.-',name='msg[0:20]')
    visualizer.print_bits(msg, p.bps)

    rx_data = simulate_comm_system(msg, p)

    visualizer.plot_constellation_map_with_points(rx_data, p)


def simulate_comm_system(msg_bits: np.ndarray, parameters):
    # msg -> [encoder] -> r[xi,0] -> [INFT] -> q[t,0]
    # -> [channel] -> q[t,L] ->
    # ->[NFT] -> r[xi,L] ->[equalizer] -> r[xi,L]_eq -> [decoder] -> msg

    modulated_data = data.data_encoder(msg_bits, parameters)  # r[xi,0] = [1+j,-1+j,1-j,...]
    visualizer.plot_constellation_map_with_points(modulated_data, parameters)

    # normalized_modulated_data = sp.normalize_vec(modulated_data, parameters)
    normalized_modulated_data = modulated_data

    # TODO: pulse shaping + up-sampling
    signal = pulse_shaping.pulse_shaping(normalized_modulated_data)
    visualizer.my_plot(np.real(signal)[0:100], name='signal=X(xi) * h(xi)', ylabel='Re{signal}')

    # TODO: divide the full signal to packages of ~100-120 symbols (try 32 / 64 symbols - on test 128)
    folded_signal = fold_subvectors(signal, N=parameters.NFT_size)
    tx_samples = []
    for sub_signal in folded_signal:
        tx_subsignal = NFT.INFT(sub_signal, parameters.Tmax)  # q[t,0]
        tx_samples.append(tx_subsignal)
    rx_samples = sp.pass_through_channel(tx_samples)  # q[t,L]
    rx_data = NFT.NFT(rx_samples, parameters)  # r[xi,L]
    rx_data_eq = sp.channel_equalizer(rx_data)

    unnormalized_rx_vec = sp.unnormalize_vec(rx_data_eq, parameters)

    return unnormalized_rx_vec


def fold_subvectors(vec: np.ndarray, N: int = 100) -> np.ndarray:
    """

    :param vec: long vector to be sliced into multiple sub vectors
    :param N: desired output sub-vector len
    :return: multi vector with length of N - each
    """
    # extend to multiplication of N
    N_zeros = int(np.ceil(len(vec) / N) * N - len(vec))
    vec = np.pad(vec,[0, N_zeros])
    # slice it
    vec = vec.reshape([-1, N])
    return vec


if __name__ == "__main__":
    main()
