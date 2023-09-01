from dataclasses import dataclass

import numpy as np
from lib.rrcos import rrcosfilter
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class SpectralShaperConfig:
    bet: float = 0.2
    Ts: float = 1
    Nos: int = 16

class SpectralShaper(Block):
    name = BlockNames.BLOCK_3_SPECTRAL_SHAPER
    def __init__(self, config: SpectralShaperConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        Ns = extra_inputs['Ns']
        bet = config.bet
        Ts = config.Ts
        fs = config.Nos/Ts
        h_ind, psi_xi = rrcosfilter(N=Ns, alpha=bet, Ts=Ts, Fs=fs)
        self.psi_t = np.fft.fft(psi_xi)  # can be saved for efficiency


    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        u_in = np.fft.ifft(np.fft.fft(x) * self.psi_t)
        self._outputs = [u_in]
        return u_in
    
    def get_output_names(self):
        return ["u_in"]

    