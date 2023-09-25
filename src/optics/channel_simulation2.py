from typing import Tuple
import numpy as np
import pyrallis
import time
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.blocks.edge_blocks.block0_input_generator import InputGenerator
from src.optics.blocks.edge_blocks.block11_evaluator import Evaluator
from src.optics.config_manager import ChannelConfig, ConfigConverter
from src.optics.blocks.block_manager import BlockManager
from src.optics.blocks import Modulator,    OverSampling,   SpectralShaper, PreEqualizer, Inft
from src.optics.blocks import Decoder,      MatchFilter,                    PostEqualizer,  Nft, Ssf


class ChannelSimulator2:
    def __init__(self, channel_config: ChannelConfig) -> None:
        self.channel_config = channel_config
        self.cb_configs = ConfigConverter.main_config_to_block_configs(channel_config)
        pack_of_blocks = self._initiate_blocks()
        self.block0: InputGenerator = pack_of_blocks[0]
        self.blocks: list = pack_of_blocks[1]
        self.block11: Evaluator = pack_of_blocks[2]
        self.io_type = channel_config.io_type

    def simulate_and_analyze(self) -> Tuple[float, int]:
        # output:
        #   ber: bit error rate (float)
        #   num_errors: number of errors (int)

        x = self._gen_input()
        extra_runtime_inputs = None  # for future use
        for i in range(len(self.blocks)):
            x = self.blocks[i].execute(x, extra_runtime_inputs)
        tx_bin = self.blocks[0].get_outputs()[0]
        rx_bin = self.blocks[-1].get_outputs()[0]
        num_errors, ber = self.block11.calc_ber(tx_bin, rx_bin)
        return ber, num_errors

    def quick_simulate(self) -> None:
        if self.channel_config.verbose:
            self._quick_simulate_verbose()
        else:
            x = self._gen_input()
            extra_runtime_inputs = None  # for future use
            for i in range(len(self.blocks)):
                x = self.blocks[i].execute(x, extra_runtime_inputs)

    def _quick_simulate_verbose(self) -> None:
        x = self._gen_input()
        extra_runtime_inputs = None  # for future use
        for i in range(len(self.blocks)):
            timestamp_before = time.time()
            x = self.blocks[i].execute(x, extra_runtime_inputs)
            dt = time.time() - timestamp_before
            block_name = self.blocks[i].name
            print(f'{dt:.2f} seconds - to evaluate block {i} ({block_name}) ')        

    def get_io_samples(self) -> Tuple[np.ndarray, np.ndarray]:

        message_s_bin, c_in             = self.blocks[0].get_outputs()
        c_in1                           = self.blocks[1].get_outputs()
        u_in, psi_xi, psi_t             = self.blocks[2].get_outputs()
        u1, b_in1, b_in, b_in_padded    = self.blocks[3].get_outputs()
        b_out_padded, b_out, is_error   = self.blocks[6].get_outputs()
        b_out1, u1_out, u_out           = self.blocks[7].get_outputs()
        c_out1, c_out                   = self.blocks[8].get_outputs()


        if self.io_type == 'b':
            x = b_out  # dirty
            y = b_in  # clean
        elif self.io_type == 'b1':
            x = b_out1  # dirty
            y = b_in1  # clean
        elif self.io_type == 'c':
            x = c_out
            y = c_in

        else:
            raise Exception(f'io_type={self.io_type} is not supported')    

        return x, y
    
    def io_to_msg(self, x: np.ndarray) -> np.ndarray:
        # roll forward the middle stage in the channel until message is reached
        
        if self.io_type == 'b':
            b =     x
            b1 =    self.blocks[7].post_compensate(b)
            b1 =    self.blocks[7].clip(b1)
            u1 =    self.blocks[7].descale(b1)
            u =     self.blocks[7].de_normalize(u1)
            u =     self.blocks[7].clear_nans(u)
            c =     self.blocks[8].execute(u, None) # match filter
        elif self.io_type == 'b1':
            b1 =    x
            b1 =    self.blocks[7].clip(b1)
            u1 =    self.blocks[7].descale(b1)
            u =     self.blocks[7].de_normalize(u1)
            u =     self.blocks[7].clear_nans(u)
            c =     self.blocks[8].execute(u, None) # match filter
        elif self.io_type == 'c':
            c =     x

        s = self.blocks[9].execute(c, None) # binary message

        return s
    
    def io_to_c_constellation(self, x: np.ndarray) -> np.ndarray:
        # roll forward the middle stage in the channel until c_out is reached (constellation point)
        if self.io_type == 'b':
            b = x
            b1 =    self.blocks[7].post_compensate(b)
            b1 =    self.blocks[7].clip(b1)
            u1 =    self.blocks[7].descale(b1)
            u =     self.blocks[7].de_normalize(u1)
            u =     self.blocks[7].clear_nans(u)
            c =     self.blocks[8].execute(u, None)
        elif self.io_type == 'b1':
            b1 = x
            b1 =    self.blocks[7].clip(b1)
            u1 =    self.blocks[7].descale(b1)
            u =     self.blocks[7].de_normalize(u1)
            u =     self.blocks[7].clear_nans(u)
            c =     self.blocks[8].execute(u, None)
        elif self.io_type == 'c':
            c = x

        return c



    def _initiate_blocks(self) -> Tuple[InputGenerator, list, Evaluator]:
        blocks = [[]] * 10
        # for block_class in BlockManager.get_channel_block_classes():
        #     block_config = ConfigConverter.fetch_config(self.configs, block_class.name)
        #     blocks.append(block_class(block_config))

        block0:     InputGenerator = InputGenerator(self.cb_configs.input_generator_config)
        blocks[0]:  Modulator = Modulator(self.cb_configs.modulator_config)
        blocks[1]:  OverSampling = OverSampling(self.cb_configs.over_sampler_config)
        blocks[2]:  SpectralShaper = SpectralShaper(self.cb_configs.spectral_shaping_config)
        blocks[3]:  PreEqualizer = PreEqualizer(self.cb_configs.pre_equalizer_config)
        blocks[4]:  Inft = Inft(self.cb_configs.inft_config)
        blocks[5]:  Ssf = Ssf(self.cb_configs.ssf_config)
        blocks[6]:  Nft = Nft(self.cb_configs.nft_config)
        blocks[7]:  PostEqualizer = PostEqualizer(self.cb_configs.post_equalizer_config)
        blocks[8]:  MatchFilter = MatchFilter(self.cb_configs.match_filter_config)
        blocks[9]:  Decoder = Decoder(self.cb_configs.decoder_config)
        block11:    Evaluator = Evaluator(self.cb_configs.evaluator_config)

        return block0, blocks, block11

    def _gen_input(self) -> np.ndarray:
        return self.block0.execute()

    def get_block(self, block_name) -> Block:
        for block in self.blocks:
            if block.name == block_name:
                return block
        return None

    def update_mu(self, mu) -> None:
        self.channel_config.mu = float(mu)        
        self.get_block(BlockNames.BLOCK_4_PRE_EQUALIZER).mu = mu
        self.get_block(BlockNames.BLOCK_8_POST_EQUALIZER).mu = mu


def sanity_test():
    config_path = './config/channel_simulation2/example.yaml'
    config = pyrallis.parse(ChannelConfig, config_path)
    cs = ChannelSimulator2(config)
    ber, num_erros = cs.simulate_and_analyze()
    print(cs.stages)


if __name__ == "__main__":
    sanity_test()
