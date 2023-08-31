from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class PreEqualizerConfig:
    mu: float = 0.1
    span_length: float = 80
    Zn: float = 0


class PreEqualizer(Block):
    name = BlockNames.BLOCK_4_PRE_EQUALIZER
    def __init__(self, config: PreEqualizerConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        self.mu = config.mu
        self.span_length = config.span_length
        Nnft = extra_inputs['Nnft']
        Ns = extra_inputs['Ns']
        self.pads = ((Nnft-Ns))//2
        self.xi = extra_inputs['xi']
        self.Zn = config.Zn

    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        u1 = self.mu*x                                                      # normalizing
        b_in1 = np.sqrt(1-np.exp(-np.abs(u1)**2))*np.exp(1j*np.angle(u1))   # #scaling
        b_in = b_in1*np.exp(-1j*self.xi**2*(self.span_length/self.Zn))      # Pre-compensation
        b_in_padded = np.pad(b_in, (self.pads, self.pads), mode='constant', constant_values=0) # padding

        self._outputs = [u1, b_in1, b_in, b_in_padded]
        return b_in_padded
