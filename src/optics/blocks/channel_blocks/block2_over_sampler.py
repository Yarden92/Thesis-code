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

    def execute(self, c_in: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        c_in1 = np.zeros(self.Ns, dtype=np.complex64)
        c_in1[::self.N_os] = c_in

        self._outputs = [c_in1]
        return c_in1
    
    def get_output_names(self):
        return ["c_in1 (over sampled c_in)"]
