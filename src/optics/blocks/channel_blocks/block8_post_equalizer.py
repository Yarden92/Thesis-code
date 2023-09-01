from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class PostEqualizerConfig:
    Zn: float = 50
    mu: float = 1
    span_length: float = 80
    with_ssf: bool = True

class PostEqualizer(Block):
    name = BlockNames.BLOCK_8_POST_EQUALIZER
    def __init__(self, config: PostEqualizerConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        self.Zn = config.Zn
        self.mu = config.mu
        self.span_length = config.span_length
        self.with_ssf = config.with_ssf
        self.xi = extra_inputs['xi']


    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        # post compensate
        if self.with_ssf:
            b_out1 = x * np.exp(-1j * self.xi**2 * (self.span_length/self.Zn))
        else:
            b_out1 = x * np.exp(1j * self.xi**2 * (self.span_length/self.Zn))

        # clip b1 to |b1| < 1
        b_out1 = np.clip(b_out1, -1, 1)

        # descale
        u1_out = np.sqrt(-np.log(1 - np.abs(b_out1)**2)) * np.exp(1j * np.angle(b_out1))

        #de-normalize
        u_out = u1_out / self.mu

        # replace nan with previous neighbors
        nan_mask = np.isnan(u_out)
        nan_indices = np.where(nan_mask)[0]
        for i in nan_indices:
            u_out[i] = u_out[i-1]

        self._outputs = [b_out1, u1_out, u_out]
        return u_out
    
    def get_output_names(self):
        return ["b_out1 (post compensated)", "u1_out (descaled)", "u_out (de-normalized)"]