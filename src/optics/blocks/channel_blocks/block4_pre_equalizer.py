from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class PreEqualizerConfig:
    mu: float = 0.1
    L: float = 1000
    Zn: float = 0
    Nnft: int = 1024
    Ns: int = 16
    xi: np.ndarray = np.array([])


class PreEqualizer(Block):
    name = BlockNames.BLOCK_4_PRE_EQUALIZER
    def __init__(self, config: PreEqualizerConfig) -> None:
        super().__init__(config)
        self.mu = config.mu
        self.L = config.L
        Nnft = config.Nnft
        Ns = config.Ns
        self.pads = ((Nnft-Ns))//2-1, ((Nnft-Ns))//2+1
        self.xi = config.xi
        self.Zn = config.Zn

    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        u1 = self.mu*x                                                      # normalizing
        b_in1 = np.sqrt(1-np.exp(-np.abs(u1)**2))*np.exp(1j*np.angle(u1))   # #scaling
        b_in = b_in1*np.exp(-1j*self.xi**2*(self.L/self.Zn))      # Pre-compensation
        b_in_padded = np.pad(b_in, self.pads, mode='constant', constant_values=0) # padding

        self._outputs = [u1, b_in1, b_in, b_in_padded]
        return b_in_padded

    def get_output_names(self):
        return ["u1 (normalized)", "b_in1 (scaled)", "b_in (pre compensated)", "b_in_padded"]