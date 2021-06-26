import numpy as np

from FNFTpy.FNFTpy import nsev_inverse, nsev, nsev_inverse_xi_wrapper
from thesis.src import params as p


def INFT(Q_xi, params):
    # TODO: validate this whole function

    # set values
    M = params.length_of_xi

    tvec = create_tvec(params)
    xivec = create_xivec(params, tvec)

    # contspec, bound_states, discspec = what_are_those(Q_xi, params)

    contspec = Q_xi
    bound_states = []
    discspec = []

    # call function
    res = nsev_inverse(xivec, tvec, contspec, bound_states, discspec)
    assert res['return_value'] == 0, "INFT failed"

    return res['q']


def NFT(q, params):
    # TODO: validate this whole function
    Xi1 = - params.BW
    Xi2 = params.BW
    M = params.length_of_xi
    tvec = create_tvec(params)
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


def create_xivec(params, tvec: np.ndarray = None):
    if tvec is None:
        tvec = create_tvec(params)
    D = params.length_of_time
    M = params.length_of_xi
    rv, xi = nsev_inverse_xi_wrapper(D, tvec[0], tvec[-1], M)
    xivec = xi[0] + np.arange(M) * (xi[1] - xi[0]) / (M - 1)
    return xivec


def create_tvec(params):
    tvec = np.linspace(-params.Tmax, params.Tmax, params.length_of_time)
    return tvec
