from abc import ABC, abstractmethod
from typing import final
import numpy as np


class Block(ABC):
    name = "Abstract format for Block"
    def __init__(self, config: object, extra_inputs: dict) -> None:
        self.config = config
        self._outputs = []

    @abstractmethod
    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        # execute the block and save all the outputs in self._output
        # returns the necessary output for the next block
        pass


    @final
    def get_outputs(self):
        if not self._outputs:
            raise Exception("Block needs to be executed first")
        return self._outputs
