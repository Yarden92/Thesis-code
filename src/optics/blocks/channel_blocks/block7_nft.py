from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_wrapper import nsev

@dataclass
class NFTConfig:
    xi_padded: np.ndarray
    t_padded: np.ndarray
    Nnft: int
    Ns: int


class Nft(Block):
    # TODO move cropping part to next stage (Equalizer)
    name = BlockNames.BLOCK_7_NFT
    def __init__(self, config: NFTConfig) -> None:
        super().__init__(config)
        self.Ns = config.Ns
        self.Nnft = config.Nnft
        self.t_padded = config.t_padded
        xi_padded = config.xi_padded
        self.XI = (xi_padded[0], xi_padded[-1])
        self.cst = 1
        self.crop_l = (self.Nnft-self.Ns)//2-1
        self.crop_r = (self.Nnft+self.Ns)//2-1


    def execute(self, x: np.ndarray, extra_runtime_inputs) -> np.ndarray:
        res = nsev(x, self.t_padded, Xi1=self.XI[0], Xi2=self.XI[1], M=self.Nnft, 
                   display_c_msg=False, cst=self.cst)
        # if res['return_value'] != 0:
        #     print(f'NFT failed with error code {res["return_value"]}')
            # raise Exception(f"nsev failed with error code {res['return_value']}")
        b_out_padded = res['cont_b']

        b_out = b_out_padded[self.crop_l:self.crop_r] # removing padding

        self._outputs = [b_out_padded, b_out, res['return_value']]
        return b_out

    def get_output_names(self):
        return ["b_out_padded", "b_out (cropped)", "nsev return value"]