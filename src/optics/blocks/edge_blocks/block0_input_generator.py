from attr import dataclass
import numpy as np

from src.optics.blocks.block_names import BlockNames


@dataclass
class InputGeneratorConfig:
    M_QAM: int = 16
    N_sc: int = 256

class InputGenerator:
    name = BlockNames.BLOCK_0_INPUT_GENERATOR
    def __init__(self, config):
        self.config = config
        self._outputs = []
        self.M_QAM = config.M_QAM
        self.N_sc = config.N_sc
        

    def execute(self) -> np.ndarray:
        message_s_int = np.random.randint(0,self.M_QAM, size=self.N_sc)
        return message_s_int
