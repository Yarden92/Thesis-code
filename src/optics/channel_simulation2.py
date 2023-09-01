import numpy as np
import pyrallis
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.blocks.edge_blocks.block0_input_generator import InputGenerator
from src.optics.blocks.edge_blocks.block11_evaluator import Evaluator
from src.optics.config_manager import ChannelConfig, ConfigConverter
from src.optics.blocks.block_manager import BlockManager


class ChannelSimulator2:
    def __init__(self, channel_config: ChannelConfig) -> None:
        self.channel_config = channel_config
        self.configs = ConfigConverter.main_config_to_block_configs(channel_config)
        self.extra_inputs = ConfigConverter._calc_extra_inputs(channel_config)
        pack_of_blocks = self._initiate_blocks()
        self.block0: InputGenerator = pack_of_blocks[0]
        self.blocks: list = pack_of_blocks[1]
        self.block11: Evaluator = pack_of_blocks[2]

    def simulate(self) -> (float, int):
        # output:
        #   ber: bit error rate (float)
        #   num_errors: number of errors (int)

        x = self._gen_input()
        extra_runtime_inputs = None # for future use
        for i in range(len(self.blocks)):
            x = self.blocks[i].execute(x, extra_runtime_inputs)
        tx_bin = self.blocks[0].get_outputs()[0]
        rx_bin = self.blocks[-1].get_outputs()[0]
        num_errors, ber = self.block11.calc_ber(tx_bin, rx_bin)
        return ber, num_errors

    def _initiate_blocks(self) -> (InputGenerator ,list, Evaluator):
        blocks = []
        for block_class in BlockManager.get_channel_block_classes():
            block_config = ConfigConverter.fetch_config(self.configs,block_class.name)
            blocks.append(block_class(block_config, self.extra_inputs))

        block0_config = ConfigConverter.fetch_config(self.configs,BlockManager.block0_class.name)
        block11_config = ConfigConverter.fetch_config(self.configs,BlockManager.block11_class.name)

        block0: InputGenerator  = BlockManager.block0_class(block0_config, self.extra_inputs)
        block11: Evaluator      = BlockManager.block11_class(block11_config, self.extra_inputs)

        return block0, blocks, block11

    def _gen_input(self) -> np.ndarray:
        return self.block0.execute()
    
    def get_block(self, block_name) -> Block:
        for block in self.blocks:
            if block.name == block_name:
                return block
        return None
    
    def update_mu(self, mu) -> None:
        self.get_block(BlockNames.BLOCK_4_PRE_EQUALIZER).mu = mu
        self.get_block(BlockNames.BLOCK_8_POST_EQUALIZER).mu = mu
        


def sanity_test():
    config_path = './config/channel_simulation2/example.yaml'
    config = pyrallis.parse(ChannelConfig, config_path)
    cs = ChannelSimulator2(config)
    ber, num_erros = cs.simulate()
    print(cs.stages)


if __name__ == "__main__":
    sanity_test()
