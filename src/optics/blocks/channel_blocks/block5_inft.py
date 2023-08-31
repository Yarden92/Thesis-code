from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse


@dataclass
class INFTConfig:
    Pn: float


class INFT(Block):
    name = BlockNames.BLOCK_5_INFT
    def __init__(self, config: INFTConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        Nnft = extra_inputs['Nnft']
        Nb = extra_inputs['Nb']
        self.xi_padded = extra_inputs['xi_padded']
        self.t_padded = extra_inputs['t_padded']
        self.bound_states = []
        self.discspec = []
        self.cst = 1  # continuous spectrum type - default is None
        self.dst = 0  # default is None
        self.crop_l = (Nnft-Nb)//2
        self.crop_r = (Nnft+Nb)//2
        self.Pn = config.Pn

    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        res = nsev_inverse(self.xi_padded, self.t_padded,
                           x,
                           self.bound_states, self.discspec, cst=self.cst, dst=self.dst)
        q_in = res['q']
        qb = q_in[self.crop_l:self.crop_r]  # removing padding
        q_p = qb*np.sqrt(self.Pn)
        self._outputs = [q_in, qb, q_p]
        return q_p