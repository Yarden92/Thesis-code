from typing import Tuple, Optional

import numpy as np

from FNFTpy import nsev_inverse, nsev, nsev_inverse_xi_wrapper


def INFT(X_xi, Tmax):
    N_xi = len(X_xi)  # (=M)
    N_time = fetch_time_and_xi_lengthes(N_xi=N_xi)[0]  # (=D)
    tvec = create_tvec(Tmax, N_time)
    xivec = create_xivec(Tmax, N_time, N_xi, tvec)

    bound_states = []  # np.array([0.7j, 1.7j])
    disc_norming_const_ana = []  # [1.0, -1.0]

    res = nsev_inverse(xivec, tvec, X_xi, bound_states, disc_norming_const_ana, cst=1, dst=0)

    assert res['return_value'] == 0, "INFT failed"

    return res['q']


def NFT(x_t, BW, Tmax):
    # TODO: validate this whole function - make sure it matches INFT
    Xi1 = -BW
    Xi2 = BW
    N_time = len(x_t)  # (=D)
    N_xi = fetch_time_and_xi_lengthes(N_time=N_time)[1]  # (=M)
    tvec = create_tvec(Tmax, N_time)
    res = nsev(x_t, tvec, Xi1, Xi2, N_xi)
    assert res['return_value'] == 0, "NFT failed"
    Q = res['cont_ref']
    return Q


def what_are_those(Q_xi, params):
    # TODO: figure out those params, and validate correctness
    # set continuous spectrum
    contspec = Q_xi

    # set discrete spectrum
    bound_states = np.array([1.0j * params.beta])
    discspec = np.array([-1.0j * params.alpha / (params.gamma + params.beta)])
    return contspec, bound_states, discspec


def create_xivec(Tmax, N_time=None, N_xi=None, tvec: np.ndarray = None):
    D, M = fetch_time_and_xi_lengthes(N_time, N_xi)
    if tvec is None:
        tvec = create_tvec(Tmax, D)
    rv, xi = nsev_inverse_xi_wrapper(D, tvec[0], tvec[-1], M)
    xivec = xi[0] + np.arange(M) * (xi[1] - xi[0]) / (M - 1)
    return xivec


def create_tvec(Tmax, N_time):
    tvec = np.linspace(-Tmax, Tmax, N_time)
    return tvec


def fetch_time_and_xi_lengthes(N_time: int = None, N_xi: int = None) -> Tuple[int,int]:
    """
    convert between length of time to length of xi and reversa
    using the relationship of: N_xi = 2*N_time
    :param N_time: Optional - if given -> filling N_xi
    :param N_xi:  Optional - if given -> filling N_time
    :return: N_time and N_xi
    """
    assert (N_time or N_xi), "at least one input is required"
    if N_time:
        N_xi = int(N_time * 2)
    if N_xi:
        N_time = int(N_xi / 2)
    return N_time, N_xi

