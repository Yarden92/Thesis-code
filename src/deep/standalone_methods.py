from torch import Tensor


def torch_to_complex_numpy(tensor: Tensor):
    np_vec = tensor.detach().numpy()
    return np_vec[:, 0] + 1j*np_vec[:, 1]
