import numpy as np

from FNFTpy import nsev_inverse, nsev, nsev_inverse_xi_wrapper
from my_files.src import params as p


def INFT(X_xi, Tmax):

    N_xi = len(X_xi)  # (=M)
    N_time = int(N_xi / 2)  # (=D)
    tvec = create_tvec(Tmax, N_time)
    xivec = create_xivec(Tmax, N_time, N_xi, tvec)

    bound_states = np.array([0.7j, 1.7j])  # TODO make those [] instead
    disc_norming_const_ana = [1.0, -1.0]

    res = nsev_inverse(xivec, tvec, X_xi, bound_states, disc_norming_const_ana, cst=1, dst=0)

    assert res['return_value'] == 0, "INFT failed"

    return res['q']


def NFT(q, params):
    # TODO: validate this whole function - make sure it matches INFT
    Xi1 = - params.BW
    Xi2 = params.BW
    M = params.length_of_xi
    tvec = create_tvec(params.Tmax, params.length_of_time)
    res = nsev(q, tvec, Xi1, Xi2, M)
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


def create_xivec(Tmax, N_time, N_xi, tvec: np.ndarray = None):
    if tvec is None:
        tvec = create_tvec(Tmax, N_time)
    D = N_time
    M = N_xi
    rv, xi = nsev_inverse_xi_wrapper(D, tvec[0], tvec[-1], M)
    xivec = xi[0] + np.arange(M) * (xi[1] - xi[0]) / (M - 1)
    return xivec


def create_tvec(Tmax, N_time):
    tvec = np.linspace(-Tmax, Tmax, N_time)
    return tvec
