from dataclasses import dataclass
import numpy as np
from src.optics.blocks.block import Block
from src.optics.blocks.block_names import BlockNames
from src.optics.split_step_fourier import SplitStepFourier


@dataclass
class SSFConfig:
    beta2: float = -21 # ps^2/km
    gamma_eff: float = 0.34 # 1/W/km
    T0: float = 0 # ps
    dz: float = 0.2 # km
    span_length: float = 80 # km
    with_ssf: bool = True
    with_noise: bool = True
    # verbose: bool = False
    Pn: float = 0# W


class SSF(Block):
    name = BlockNames.BLOCK_6_SSF
    def __init__(self, config: SSFConfig, extra_inputs: dict) -> None:
        super().__init__(config, extra_inputs)
        beta2 = config.beta2
        gamma_eff = config.gamma_eff
        T0 = config.T0
        dt = extra_inputs['dt']
        dz = config.dz
        span_length = config.span_length
        with_noise = config.with_noise
        Nnft = extra_inputs['Nnft']
        Nb = extra_inputs['Nb']
        # verbose = config.verbose
        self.with_ssf = config.with_ssf
        self.Pn = config.Pn
        self.pads = ((Nnft-Nb))//2

        self.ssf = SplitStepFourier(
            b2=beta2,
            gamma=gamma_eff,
            t0=T0,
            dt=dt,
            z_n=span_length,
            dz=dz,
            with_noise=with_noise,
            # verbose=verbose,
        )

    def execute(self, x: np.ndarray, extra_inputs) -> np.ndarray:
        if self.with_ssf:
            qz = self.ssf(x)
        else:
            qz = x

        q_s = qz/np.sqrt(self.Pn) # rescale to soliton units
        q_pad = np.pad(q_s, (self.pads, self.pads), mode='constant', constant_values=0) # pad to length of N_NFT

        self._outputs = [qz, q_s, q_pad]
        return q_pad

    def get_output_names(self):
        return ["qz (after ssf)", "q_s (soliton units)", "q_pad (padded)"]