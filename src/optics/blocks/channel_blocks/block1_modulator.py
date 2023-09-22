from dataclasses import dataclass
import numpy as np
from src.general_methods.signal_processing import SP
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from ModulationPy import ModulationPy


@dataclass
class ModulatorConfig:
    M_QAM: int = 16


class Modulator(Block):
    name = BlockNames.BLOCK_1_MODULATOR
    def __init__(self, config: ModulatorConfig) -> None:
        super().__init__(config)
        self.M_QAM = config.M_QAM
        self.sps = int(np.log2(self.M_QAM))
        self.modem = ModulationPy.QAMModem(self.M_QAM, soft_decision=False)



    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        # input:
        #   x: np array (length: N_sc) of integers in range [0, M_QAM)
        
        message_s_bin = SP.unpackbits(x, self.sps)
        c_in = self.modem.modulate(message_s_bin)

        self._outputs = [message_s_bin, c_in]
        return c_in
    
    def get_output_names(self):
        return ["message_s_bin", "c_in"]
