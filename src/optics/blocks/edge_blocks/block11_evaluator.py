from dataclasses import dataclass
import numpy as np

from src.optics.blocks.block_names import BlockNames

@dataclass
class EvaluatorConfig:
    M_QAM: int = 16
    N_sc: int = 256
    verbose: bool = False

class Evaluator:
    name = BlockNames.BLOCK_11_EVALUATOR
    def __init__(self, config: EvaluatorConfig, extra_inputs: dict) -> None:
        M_QAM = config.M_QAM
        self.N_sc = config.N_sc
        self.N_bin = self.N_sc * np.log2(M_QAM)
        self.verbose = config.verbose

    def calc_ber(self, x: np.ndarray, y: np.ndarray) -> (int, float):
        # inputs:
            # two binary arrays of the same length
        num_errors = (x != y).sum()
        ber = num_errors / self.N_bin

        if self.verbose:
            print(f'ber = {ber} = {num_errors}/{self.N_bin}')

        return num_errors, ber

    def calc_ser(self, x: np.ndarray, y: np.ndarray) -> float:
        # inputs:
            # two integer arrays of the same length
        num_errors = (x != y).sum()
        ser = num_errors / self.N_sc

        if self.verbose:
            print(f'ser = {ser} = {num_errors}/{self.N_sc}')

        return num_errors, ser