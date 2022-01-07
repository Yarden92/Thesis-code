from src.channel_simulation import *
from src.output import Outputs
from src.params import Params


def main():
    p = Params(m_qam=16,
           num_symbols=64,
           normalization_factor=1,
           plot_vec_after_creation=False)
    ber, num_errors = iterate_through_channel(p)

    print(f'ber = {ber}, N_errors = {num_errors}')

def iterate_through_channel(par):
    o2 = Outputs()
    o2.x[0] = Step0.generate_message(par)
    o2.x[1], o2.modem = Step1.modulate(o2.x[0], par)
    o2.x[2] = Step2.over_sample(o2.x[1], par)
    o2.x[3], o2.h_rrc, o2.L_rrc = Step3.pulse_shape(o2.x[2], par)
    o2.x[4] = Step4.pre_equalize(o2.x[3], par)
    Step5.add_nft_params(o2.x[4], par, o2)
    o2.x[5] = Step5.inft(o2.x[4], par, o2)
    o2.x[6] = Step6.channel(o2.x[5], par)
    o2.x[7] = Step7.nft(o2.x[6], par, o2)
    o2.x[8] = Step8.equalizer(o2.x[7], par)
    o2.x[9] = Step9.match_filter(o2.x[8], par, o2)
    o2.x[10], ber, num_errors = Step10.demodulate(o2.x[9], par, o2, o2.x[0], o2.modem)
    return ber, num_errors

if __name__ == '__main__':
    main()