from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from ModulationPy import ModulationPy


@dataclass
class DecoderConfig:
    M_QAM: int = 16


class Decoder(Block):
    name = BlockNames.BLOCK_10_DECODER
    def __init__(self, config: DecoderConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        self.M_QAM = config.M_QAM
        self.sps = int(np.log2(self.M_QAM))
        self.modem = ModulationPy.QAMModem(self.M_QAM, soft_decision=False)

    def execute(self, x: np.ndarray, extra_inputs) -> None:
        s_out = self.modem.demodulate(x)
        # TODO: pack bits to integers: SP.packbits(s_out, self.sps)

        self._outputs = [s_out]
        return s_out
    
    def get_output_names(self):
        return ["s_out (binary)"]