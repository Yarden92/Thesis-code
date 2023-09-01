from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_wrapper import nsev

@dataclass
class NFTConfig:
    pass


class NFT(Block):
    name = BlockNames.BLOCK_7_NFT
    def __init__(self, config: NFTConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        self.t_padded = extra_inputs['t_padded']
        xi_padded = extra_inputs['xi_padded']
        self.XI = (xi_padded[0], xi_padded[-1])
        self.Nnft = extra_inputs['Nnft']
        self.Ns = extra_inputs['Ns']
        self.cst = 1
        self.crop_l = (self.Nnft-self.Ns)//2
        self.crop_r = (self.Nnft+self.Ns)//2


    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        res = nsev(x, 
                   self.t_padded, Xi1=self.XI[0], Xi2=self.XI[1], M=self.Nnft, 
                   display_c_msg=False, cst=self.cst)
        assert res['return_value'] == 0, "NFT failed"
        b_out_padded = res['cont_b']

        b_out = b_out_padded[self.crop_l:self.crop_r] # removing padding

        self._outputs = [b_out_padded, b_out]
        return b_out

    def get_output_names(self):
        return ["b_out_padded", "b_out (cropped)"]