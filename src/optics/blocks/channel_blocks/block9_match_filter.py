from dataclasses import dataclass
import numpy as np
from lib.rrcos import rrcosfilter
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class MatchFilterConfig:
    Nos: int = 16
    bet: float = 0.2
    Ts: float = 1


class MatchFilter(Block):
    name = BlockNames.BLOCK_9_MATCH_FILTER
    def __init__(self, config: MatchFilterConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        Ns = extra_inputs['Ns']
        bet = config.bet
        Ts = config.Ts
        self.Nos = config.Nos
        fs = self.Nos/Ts
        self.start = 0
        self.stop = Ns
        self.step = self.Nos
        h_ind, psi_xi = rrcosfilter(N=Ns, alpha=bet, Ts=Ts, Fs=fs)
        self.psi_t = np.fft.fft(psi_xi)  # can be saved for efficiency

    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        c_out1 = np.fft.ifft(self.psi_t*np.fft.fft(x)) / self.Nos  # convolve with RRC again
        c_out = c_out1[self.start:self.stop:self.step]  # downsample

        self._outputs = [c_out1, c_out]
        return c_out
