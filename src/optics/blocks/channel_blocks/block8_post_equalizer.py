from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class PostEqualizerConfig:
    Zn: float = 50
    mu: float = 1
    L: float = 1000
    with_ssf: bool = True

class PostEqualizer(Block):
    name = BlockNames.BLOCK_8_POST_EQUALIZER
    def __init__(self, config: PostEqualizerConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        self.Zn = config.Zn
        self.mu = config.mu
        self.L = config.L
        self.with_ssf = config.with_ssf
        self.xi = extra_inputs['xi']


    def execute(self, b: np.ndarray, extra_inputs, ) -> np.ndarray:
        b_out1 = self.post_compensate(b)
        b_out1 = self.clip(b_out1)
        u1_out = self.descale(b_out1)
        u_out = self.de_normalize(u1_out)
        u_out = self.clear_nans(u_out)

        self._outputs = [b_out1, u1_out, u_out]
        return u_out
    
    def get_output_names(self):
        return ["b_out1 (post compensated)", "u1_out (descaled)", "u_out (de-normalized)"]
    
    def post_compensate(self, b: np.ndarray) -> np.ndarray:
        if self.with_ssf:
            b1 = b * np.exp(-1j * self.xi**2 * (self.L/self.Zn))
        else:
            b1 = b * np.exp(1j * self.xi**2 * (self.L/self.Zn))

        return b1
    
    def clip(self, b1: np.ndarray) -> np.ndarray:
        # clip b1 to |b1| < 1
        max_val = np.max(np.abs(b1))
        if max_val >= 1:
            b1 = b1 / max_val * 0.999999
        return b1

    def descale(self, b1: np.ndarray) -> np.ndarray:
        u1 = np.sqrt(-np.log(1 - np.abs(b1)**2)) * np.exp(1j * np.angle(b1))
        return u1

    def de_normalize(self, u1: np.ndarray) -> np.ndarray:
        u = u1 / self.mu
        return u
    
    def clear_nans(self, u: np.ndarray) -> np.ndarray:
        # replace nan with previous neighbors
        nan_mask = np.isnan(u)
        nan_indices = np.where(nan_mask)[0]
        for i in nan_indices:
            u[i] = u[i-1]
        return u
