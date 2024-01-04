from dataclasses import dataclass
from re import L
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.split_step_fourier import SplitStepFourier


@dataclass
class SSFConfig:
    beta2: float = -21 # ps^2/km
    gamma: float = 0.34 # 1/W/km
    dz: float = 0.2 # km
    K_T: float = 1.1
    chi: float = 0.0461
    L: float = 9600     # km
    with_ssf: bool = True
    with_noise: bool = True
    # verbose: bool = False
    Pn: float = 0   # W
    dt: float = 0.2 # ps
    Nnft: int = 1024
    Nb: int = 16


class Ssf(Block):
    name = BlockNames.BLOCK_6_SSF
    def __init__(self, config: SSFConfig) -> None:
        super().__init__(config)
        dt = config.dt
        Nnft = config.Nnft
        Nb = config.Nb
        # verbose = config.verbose
        self.with_ssf = config.with_ssf
        self.Pn = config.Pn
        self.pads = ((Nnft-Nb))//2

        self.ssf = SplitStepFourier(
            b2=config.beta2,
            gamma=config.gamma,
            dt=dt,
            L=config.L,
            Nt=Nb,
            K_T=config.K_T,
            chi=config.chi,
            dz=config.dz,
            with_noise=config.with_noise,
            # verbose=verbose,
        )

    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        if self.with_ssf:
            qz = self.ssf(x)
        else:
            qz = x

        q_s = qz/np.sqrt(self.Pn) # rescale to soliton units
        q_pad = np.pad(q_s, (self.pads, self.pads), mode='constant', constant_values=0) # pad to length of N_NFT

        self._outputs = [qz, q_s, q_pad]
        return q_pad

    def get_output_names(self):
        return ["qz (after ssf)", "q_s (soliton units)", "q_pad (padded)"]