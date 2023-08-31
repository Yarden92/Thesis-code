import numpy as np
import pyrallis
from src.optics.config_manager import MainConfig, ConfigConverter
from src.optics.blocks.block_manager import BlockManager


class ChannelSimulator2:
    def __init__(self, main_config: MainConfig) -> None:
        self.configs = ConfigConverter.main_config_to_block_configs(main_config)
        self.extra_inputs = ConfigConverter._calc_extra_inputs(main_config)
        self.block0, self.blocks, self.block11 = self._initiate_blocks()

    def simulate(self) -> None:
        x = self._gen_input()
        extra_runtime_inputs = None # for future use
        for i in range(len(self.blocks)):
            x = self.blocks[i].execute(x, extra_runtime_inputs)

    def _initiate_blocks(self) -> None:
        blocks = []
        for block_class in BlockManager.get_channel_block_classes():
            block_config = ConfigConverter.fetch_config(self.configs,block_class.name)
            blocks.append(block_class(block_config, self.extra_inputs))

        block0_config = ConfigConverter.fetch_config(self.configs,BlockManager.block0_class.name)
        block0 = BlockManager.block0_class(block0_config)

        block11_config = ConfigConverter.fetch_config(self.configs,BlockManager.block11_class.name)
        block11 = BlockManager.block11_class(block11_config, self.extra_inputs)

        return block0, blocks, block11

    def _gen_input(self) -> np.ndarray:
        return self.block0.execute()


def sanity_test():
    config_path = './config/channel_simulation2/example.yaml'
    config = pyrallis.parse(MainConfig, config_path)
    cs = ChannelSimulator2(config)
    cs.simulate()
    print(cs.stages)


if __name__ == "__main__":
    sanity_test()
