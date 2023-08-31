from dataclasses import dataclass, field
import numpy as np
from src.optics.blocks.block_names import BlockNames
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse_xi_wrapper
from src.optics.blocks.edge_blocks.block0_input_generator import InputGeneratorConfig
from src.optics.blocks.channel_blocks.block1_modulator import ModulatorConfig
from src.optics.blocks.channel_blocks.block2_over_sampler import OverSamplingConfig
from src.optics.blocks.channel_blocks.block3_spectral_shaper import SpectralShaperConfig
from src.optics.blocks.channel_blocks.block4_pre_equalizer import PreEqualizerConfig
from src.optics.blocks.channel_blocks.block5_inft import INFTConfig
from src.optics.blocks.channel_blocks.block6_ssf import SSFConfig
from src.optics.blocks.channel_blocks.block7_nft import NFTConfig
from src.optics.blocks.channel_blocks.block8_post_equalizer import PostEqualizerConfig
from src.optics.blocks.channel_blocks.block9_match_filter import MatchFilterConfig
from src.optics.blocks.channel_blocks.block10_decoder import DecoderConfig
from src.optics.blocks.edge_blocks.block11_evaluator import EvaluatorConfig


@dataclass
class MainConfig:

    # System Configuration:
    W: float = 0.05             # Total bandwidth, estimated [THz]
    Nspans: int = 12            # The number of spans
    span_length: int = 80       # Transmission span [km]
    M_QAM: int = 64             # QAM order (2,4,16,64,256)
    Ts: float = 1               # Symbol period [??]

    # Modulation and Coding:
    Nos: int = 16               # Oversampling factor (must be even)
    eta: int = 2                # spectral efficiency penalty factor (1,4]
    mu: float = 0.1             # Dimensionless power scaling factor (RRC)
    bet: float = 0.2            # roll-off factor
    with_ssf: bool = True       # whether to use SSF or not
    with_noise: bool = True     # whether to add noise or not

    # Fiber and Dispersion:
    alphadB: float = 0.2        # Power loss db/km
    beta2: float = -21          # ps^2/km
    gamma: float = 1.27         # Nonlinear coefficient in [1/km*W]
    dz: float = 0.2             # Z-step, [km] - initial step estimate

    def __post_init__(self):
        # direct calculations
        self.total_propagation_distance = self.span_length*self.Nspans  # km
        self.alpha = self.alphadB*np.log(10)/10                         # Power loss [1/km]

        # calculations
        self.T_guardband = 1.5*np.pi*self.W*abs(self.beta2)*self.total_propagation_distance  # guard band [ps] + 50%
        self.N_sc_raw = self.W*self.T_guardband / (self.eta-1)  # The required number of subcarriers
        self.N_sc = int(2**np.ceil(np.log2(self.N_sc_raw)))     # rounded to nearest factor of two
        self.T0 = self.N_sc/self.W                              # Useful symbol duration, normalization time [ps]
        self.Tb = self.T0*self.eta                              # Burst width
        self.G = np.exp(self.alpha*self.span_length)            # The power gain factor
        if self.G > 1:
            self.gamma_eff = self.gamma*(self.G-1)/(self.G*np.log(self.G))

        self.Tn = self.T0/(np.pi*(1+self.bet))             # [ps]
        self.Zn = self.Tn**2/abs(self.beta2)               # [km]
        self.Pn = 1/(self.gamma_eff*self.Zn)               # [W]

        # self.Nsps = np.ceil(self.span_length/self.dz)               # Required number of steps per span


@dataclass
class ChannelConfig:
    # THESE NAMES MUST BE ALIGNED WITH BLOCK NAMES
    input_generator_config: InputGeneratorConfig = field(default_factory=InputGeneratorConfig)
    modulator_config: ModulatorConfig = field(default_factory=ModulatorConfig)
    over_sampler_config: OverSamplingConfig = field(default_factory=OverSamplingConfig)
    spectral_shaping_config: SpectralShaperConfig = field(default_factory=SpectralShaperConfig)
    pre_equalizer_config: PreEqualizerConfig = field(default_factory=PreEqualizerConfig)
    inft_config: INFTConfig = field(default_factory=INFTConfig)
    ssf_config: SSFConfig = field(default_factory=SSFConfig)
    nft_config: NFTConfig = field(default_factory=NFTConfig)
    post_equalizer_config: PostEqualizerConfig = field(default_factory=PostEqualizerConfig)
    match_filter_config: MatchFilterConfig = field(default_factory=MatchFilterConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    evaluator_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)


class ConfigConverter:

    @staticmethod
    def main_config_to_block_configs(config: MainConfig):
        input_config = InputGeneratorConfig(
            M_QAM=config.M_QAM,
            N_sc=config.N_sc,
        )

        modulator_config = ModulatorConfig(
            M_QAM=config.M_QAM,
        )

        over_sampler_config = OverSamplingConfig(
            N_os=config.Nos
        )

        spectral_shaping_config = SpectralShaperConfig(
            bet=config.bet,
            Ts=config.Ts,
            Nos=config.Nos,
        )

        pre_equalizer_config = PreEqualizerConfig(
            mu=config.mu,
            span_length=config.span_length,
            Zn=config.Zn,
        )

        inft_config = INFTConfig(
            Pn=config.Pn
        )

        ssf_config = SSFConfig(
            beta2=config.beta2,
            gamma_eff=config.gamma_eff,
            T0=config.T0,
            dz=config.dz,
            span_length=config.span_length,
            with_ssf=config.with_ssf,
            with_noise=config.with_noise,
            Pn=config.Pn,
        )

        nft_config = NFTConfig()

        post_equalizer_config = PostEqualizerConfig(
            Zn=config.Zn,
            mu=config.mu,
            span_length=config.span_length,
            with_ssf=config.with_ssf,
        )

        match_filter_config = MatchFilterConfig(
            Nos=config.Nos,
            bet=config.bet,
            Ts=config.Ts,
        )

        decoder_config = DecoderConfig(
            M_QAM=config.M_QAM,
        )

        evaluator_config = EvaluatorConfig(
            M_QAM=config.M_QAM,
            N_sc=config.N_sc,
        )

        return ChannelConfig(
            input_generator_config=input_config,
            modulator_config=modulator_config,
            over_sampler_config=over_sampler_config,
            spectral_shaping_config=spectral_shaping_config,
            pre_equalizer_config=pre_equalizer_config,
            inft_config=inft_config,
            ssf_config=ssf_config,
            nft_config=nft_config,
            post_equalizer_config=post_equalizer_config,
            match_filter_config=match_filter_config,
            decoder_config=decoder_config,
            evaluator_config=evaluator_config,
        )

    @staticmethod
    def _calc_extra_inputs(config: MainConfig):
        # axes
        Ns = config.N_sc*config.Nos                           # The number of meaningful points
        Nnft = int(4*2**(np.ceil(np.log2(Ns))))              # The number of points for NFT - round up to a power of 2
        Tnft = np.pi*config.Nos*config.Tn                                 # The NFT base
        dt = Tnft/Nnft                                      # The time step in ps
        Nb = config.Tb/dt                                          # The number of points in each burst
        Nb = 2*round(Nb/2)                                  # Make sure it is even

        T1 = dt*(-Nnft/2)/config.Tn
        T2 = dt*(Nnft/2-1)/config.Tn
        _, XI = nsev_inverse_xi_wrapper(Nnft, T1, T2, Nnft, display_c_msg=True)
        xi = np.arange(-Ns/2, Ns/2)/config.Nos             # Array of upsampled nonlinear frequencies
        xi_padded = np.linspace(XI[0], XI[1], Nnft)   # Array of padded nonlinear frequencies (for NFT)
        t = np.arange(-Nb/2, Nb/2)*dt
        t_padded = np.linspace(T1, T2, Nnft)

        extra_inputs = {
            'xi': xi,
            'xi_padded': xi_padded,
            't': t,
            't_padded': t_padded,
            'Nnft': Nnft,
            'Nb': Nb,
            'Ns': Ns,
            'dt': dt,

        }

        return extra_inputs

    @staticmethod
    def fetch_config(configs: ChannelConfig,name):
        if name==BlockNames.BLOCK_0_INPUT_GENERATOR:
            return configs.input_generator_config
        elif name==BlockNames.BLOCK_1_MODULATOR:
            return configs.modulator_config
        elif name==BlockNames.BLOCK_2_OVER_SAMPLER:
            return configs.over_sampler_config
        elif name==BlockNames.BLOCK_3_SPECTRAL_SHAPER:
            return configs.spectral_shaping_config
        elif name==BlockNames.BLOCK_4_PRE_EQUALIZER:
            return configs.pre_equalizer_config
        elif name==BlockNames.BLOCK_5_INFT:
            return configs.inft_config
        elif name==BlockNames.BLOCK_6_SSF:
            return configs.ssf_config
        elif name==BlockNames.BLOCK_7_NFT:
            return configs.nft_config
        elif name==BlockNames.BLOCK_8_POST_EQUALIZER:
            return configs.post_equalizer_config
        elif name==BlockNames.BLOCK_9_MATCH_FILTER:
            return configs.match_filter_config
        elif name==BlockNames.BLOCK_10_DECODER:
            return configs.decoder_config
        elif name==BlockNames.BLOCK_11_EVALUATOR:
            return configs.evaluator_config
        else:
            raise Exception("No such block name")
        