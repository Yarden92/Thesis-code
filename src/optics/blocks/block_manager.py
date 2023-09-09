from src.optics.blocks.edge_blocks.block0_input_generator import InputGenerator
from src.optics.blocks.channel_blocks.block1_modulator import Modulator
from src.optics.blocks.channel_blocks.block2_over_sampler import OverSampling
from src.optics.blocks.channel_blocks.block3_spectral_shaper import SpectralShaper
from src.optics.blocks.channel_blocks.block4_pre_equalizer import PreEqualizer
from src.optics.blocks.channel_blocks.block5_inft import Inft
from src.optics.blocks.channel_blocks.block6_ssf import Ssf
from src.optics.blocks.channel_blocks.block7_nft import Nft
from src.optics.blocks.channel_blocks.block8_post_equalizer import PostEqualizer
from src.optics.blocks.channel_blocks.block9_match_filter import MatchFilter
from src.optics.blocks.channel_blocks.block10_decoder import Decoder
from src.optics.blocks.edge_blocks.block11_evaluator import Evaluator


class BlockManager:
    block0_class = InputGenerator
    block11_class = Evaluator
    @staticmethod
    def get_channel_block_classes() -> list:
        blocks = []

        blocks.append(Modulator)
        blocks.append(OverSampling)
        blocks.append(SpectralShaper)
        blocks.append(PreEqualizer)
        blocks.append(Inft)
        blocks.append(Ssf)
        blocks.append(Nft)
        blocks.append(PostEqualizer)
        blocks.append(MatchFilter)
        blocks.append(Decoder)
        

        return blocks

    @staticmethod
    def get_block_class(name: str):
        for block_class in BlockManager.get_channel_block_classes():
            if block_class.name == name:
                return block_class
        raise Exception("Block with name: {} not found".format(name))

    
