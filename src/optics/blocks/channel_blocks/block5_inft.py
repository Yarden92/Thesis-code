from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse


@dataclass
class INFTConfig:
    Pn: float
    Nb: int
    Nnft: int
    xi_padded: np.ndarray
    t_padded: np.ndarray


class Inft(Block):
    name = BlockNames.BLOCK_5_INFT

    def __init__(self, config: INFTConfig) -> None:
        super().__init__(config)
        Nnft = config.Nnft
        Nb = config.Nb
        self.xi_padded = config.xi_padded
        self.t_padded = config.t_padded
        self.bound_states = []
        self.discspec = []
        self.cst = 1  # mistakenly 2 is actually equivalent to b of xi (rather than 1 as described)
        self.dst = 0  # default is None
        self.crop_l = (Nnft-Nb)//2
        self.crop_r = (Nnft+Nb)//2
        self.Pn = config.Pn

    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        res = nsev_inverse(self.xi_padded, self.t_padded,
                           x,
                           self.bound_states, self.discspec, cst=self.cst, dst=self.dst,
                           display_c_msg=False)
        q_in = res['q']
        qb = q_in[self.crop_l:self.crop_r]  # removing padding
        q_p = qb*np.sqrt(self.Pn)
        self._outputs = [q_in, qb, q_p]
        return q_p

    def get_output_names(self):
        return ["q_in (after inft)", "qb (cropped)", "q_p (physical units)"]
