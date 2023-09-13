from typing import Any

import numpy as np

from src.optics.myFNFTpy.FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse_xi_wrapper


class NFDMSimulator:
    def __init__(self) -> None:
        self._generate_system_parameters()
        self._amplification_scheme()
        self._modulation_constellation()
        self._carrier_characteristics()
        self._normalized_units()
        self._set_time_and_frequency_arrays()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._create_symbols_and_bursts()
        self._INFT_Tx()
        self._Split_step()
        self._FNFT_Rx()
        self._Demodulation()
        self._Constellation_diagram()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ private functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _generate_system_parameters(self) -> None:
        self.Nbursts = 1                        # The number of bursts
        self.Nspans = 12                        # The number of spans
        self.La = 12*80/self.Nspans             # Transmission span [km]
        self.beta2 = -21.0                      # GVD coefficient [ps^2/km]
        self.gamma = 1.27                       # Nonlinear coefficient in [1/km*W]
        self.eta = 2                            # SE penalty. eta=(T0+TG)/T0
        self.W = 0.05                           # Total bandwidth, estimated [Thz]
        self.TG = self._calc_Tg()               # % The required GB [ps]
        self.Nsc = self._calc_Nsc()             # The number for subcarriers
        self.Nsc = 2*(round(self.Nsc/2))        # Round it up to the next even number
        self.T0 = self.Nsc/self.W               # Useful symbol duration, normalization time [ps]
        # self.Tb=T0+TG                         # Burst width [ps]
        self.Tb = self.T0*self.eta              # Burst width
        self.no_ss = True                       # If true runs B2B run - no split step
        self.noise = False                      # If true run simulation with noise
        self.Nghost = 2                         # The number of ghost symbols inserted from each side

        def _calc_Tg(self) -> float:
            return 1.5*np.pi*self.W*abs(self.beta2)*self.La*self.Nspans

        def _calc_Nsc(self) -> float:
            return round(self.W*self.TG/(self.eta-1))

    def _amplification_scheme(self) -> None:
        self.lumped = True                      # If true use lumped amplification
        if not self.lumped:
            self.Nspans = 1
        self.dz = 0.2                           # Z-step, [km] - initial step estimate
        self.Nsps = np.ceil(self.La/self.dz)    # Required number of steps per span
        self.dz = self.La/self.Nsps             # Adjusting step slightly
        self.L = self.La*self.Nspans            # Total propagation distance
        self.alphadB = 0.2                      # Power loss db/km

        # Linear loss
        if self.lumped:
            self.alpha = self.alphadB*np.log(10)/10
        else:
            self.alpha = 0
        self.G = np.exp(self.alpha*self.La)     # The power gain factor
        self.gamma_eff = self.gamma             # The effective gamma-coefficient in r.w.u.
        if self.lumped:
            if self.G > 1:
                self.gamma_eff = self.gamma*(self.G-1)/(self.G*np.log(self.G))

    def _modulation_constellation(self) -> None:
        self.mu = 0.1                           # Dimensionless power scaling factor (RRC)
        self.M = 32                             # The order of M-QAM

    def _carrier_characteristics(self) -> None:
        self.bet = 0.2                          # The roll-off factor

    def _normalized_units(self) -> None:
        self.Tn = self.T0/(np.pi*(1+self.bet))  # [ps]
        self.Zn = self.Tn**2/abs(self.beta2)    # [km]
        self.Pn = 1/(self.gamma_eff*self.Zn)    # [W]

    def _set_time_and_frequency_arrays(self) -> None:
        # matlab version
        # Nos=16; % Oversampling factor (must be even)
        # Ns=(Nsc+2*Nghost)*Nos; % The number of meaningful points
        # Nnft=4*2^(ceil(log2(Ns))); % The number of points for NFT - round up to a power of 2 
        # % Sometimes additional increase in #points is needed!
        # Tnft=pi*Nos*Tn; % The NFT base
        # dt=Tnft/Nnft; % The time step in ps
        # Nb=2*round(Tb/(2*dt)); % The number of points in each burst
        # tau=dt*[-Nnft/2,Nnft/2-1]/Tn;
        # [XI, xi1] = mex_fnft_nsev_inverse_XI(Nnft, tau, Nnft);
        # xi=(-Ns/2:Ns/2-1)/Nos; % Array of upsampled nonlinear frequencies

        # python version
        self.Nos = 16                                   # Oversampling factor (must be even)
        self.Ns = (self.Nsc+2*self.Nghost)*self.Nos     # The number of meaningful points
        self.Nnft = 4*2**(np.ceil(np.log2(self.Ns)))    # The number of points for NFT - round up to a power of 2
        # Sometimes additional increase in #points is needed!
        self.Tnft = np.pi*self.Nos*self.Tn              # The NFT base
        self.dt = self.Tnft/self.Nnft                   # The time step in ps
        self.Nb = 2*round(self.Tb/(2*self.dt))          # The number of points in each burst
        T1 = self.dt*(-self.Nnft/2)/self.Tn
        T2 = self.dt*(self.Nnft/2-1)/self.Tn
        self.XI, self.xi1 = self._mex_fnft_nsev_inverse_XI(self.Nnft, T1,T2, self.Nnft)
        self.xi = np.arange(-self.Ns/2, self.Ns/2-1)/self.Nos  # Array of upsampled nonlinear frequencies

    def _mex_fnft_nsev_inverse_XI(self, Nnft: int, T1: float,T2: float) -> np.ndarray:
        rv, xi = nsev_inverse_xi_wrapper(
            D=Nnft,
            T1=T1,
            T2=T2,
            M=Nnft,
            dis=None,
            display_c_msg=True)
