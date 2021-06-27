import numpy as np

from thesis.src import NFT
from thesis.src import pulse_shaping
from thesis.src import visualizer
from thesis.src import params as p
from thesis.src import data
from thesis.src import signal_processing as sp


def main():
    msg = data.generate_random_msg(p.length_of_msg)  # binary vec [0,1,0,...]
    rx_data = simulate_comm_system(msg, p)

    visualizer.plot_constellation_map_with_points(rx_data, p)


def simulate_comm_system(msg_bits: np.ndarray, parameters):
    # msg -> [encoder] -> r[xi,0] -> [INFT] -> q[t,0]
    # -> [channel] -> q[t,L] ->
    # ->[NFT] -> r[xi,L] ->[equalizer] -> r[xi,L]_eq -> [decoder] -> msg

    modulated_data = data.data_encoder(msg_bits, parameters)  # r[xi,0] = [1+j,-1+j,1-j,...]

    # normalized_modulated_data = sp.normalize_vec(modulated_data, parameters)
    normalized_modulated_data = modulated_data

    # TODO: pulse shaping + upsampling
    signal = pulse_shaping.pulse_shaping(normalized_modulated_data)

    tx_samples = NFT.INFT(signal, parameters)  # q[t,0]
    rx_samples = sp.pass_through_channel(tx_samples)  # q[t,L]
    rx_data = NFT.NFT(rx_samples, parameters)  # r[xi,L]
    rx_data_eq = sp.channel_equalizer(rx_data)

    unnormalized_rx_vec = sp.unnormalize_vec(rx_data_eq, parameters)

    return unnormalized_rx_vec


if __name__ == "__main__":
    main()
