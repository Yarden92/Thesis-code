from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames


@dataclass
class OverSamplingConfig:
    N_os: int = 16
    Ns: int = 16


class OverSampling(Block):
    name = BlockNames.BLOCK_2_OVER_SAMPLER
    def __init__(self, config: OverSamplingConfig) -> None:
        super().__init__(config)
        self.N_os = config.N_os
        self.Ns = config.Ns

    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        y = np.zeros(self.Ns, dtype=np.complex64)
        y[::self.N_os] = x

        self._outputs = [y]
        return y
    
    def get_output_names(self):
        return ["c_in1 (over sampled c_in)"]
