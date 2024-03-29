import numpy as np


class SP:
    @staticmethod
    def signal_power(x: np.ndarray) -> float:
        return np.linalg.norm(x, ord=2)

    @staticmethod
    def signal_power_dbm(x: np.ndarray, dt, Tb) -> float:
        return 10*np.log10(np.sum(np.abs(x)**2)*dt/Tb/1e-3)

    @staticmethod
    def peak(x: np.ndarray) -> float:
        return np.max(np.abs(x))

    @staticmethod
    def unpackbits(x, num_bits: int):
        # x is a numpy array with N numbers of base M
        # num_bits = log2(M)
        # returns a binary numpy array with shape (N, num_bits)
        assert num_bits <= 8, "limited to uint8, otherwise change the code"
        binary_vec = np.unpackbits(x.astype(np.uint8))  # unpack into 8 bits vectors
        binary_matrix_8bits = binary_vec.reshape(-1, 8)  # fold into matrix
        binary_matrix = binary_matrix_8bits[:, -num_bits:]  # take last num_bits columns
        binary_vec = binary_matrix.reshape(-1)  # unfold into vector
        return binary_vec
    
    @staticmethod
    def calc_avg_dist(x: np.ndarray, y: np.ndarray) -> float:
        # calculate average euclidean distance between x and y
        avg_dist = np.mean(np.abs(x - y))
        return avg_dist


