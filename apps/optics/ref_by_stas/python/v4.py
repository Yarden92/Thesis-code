import numpy as np
from apps.optics.ref_by_stas.b_simulator_rrc import rrc_impulse
from src.general_methods.visualizer import Visualizer
from src.optics.myFNFTpy.FNFTpy.fnft_nsev_inverse_wrapper import nsev_inverse_wrapper, nsev_inverse_xi_wrapper
from IPython.display import display, Markdown
from ModulationPy.ModulationPy import QAMModem

# generate system parameters
Nbursts = 1                             # The number of bursts
Nspans = 12                             # The number of spans
La = 12*80/Nspans                       # Transmission span [km]
beta2 = -21.0                           # GVD coefficient [ps^2/km]
gamma = 1.27                            # Nonlinear coefficient in [1/km*W]
eta = 2                                 # SE penalty. eta=(T0+TG)/T0
W = 0.05                                # Total bandwidth, estimated [Thz]
TG = 1.5*np.pi*W*abs(beta2)*La*Nspans   # The required GB [ps]
Nsc = W*TG/(eta-1)                      # The number for subcarriers
Nsc = 2*(round(Nsc/2))                  # Round it up to the next even number
Nsc = int(2**np.ceil(np.log2(Nsc)))     # Round it up to the next power of 2
T0 = Nsc/W                              # Useful symbol duration, normalization time [ps]
Tb = T0*eta                             # Burst width
no_ss = True                            # If true runs B2B run - no split step
noise = False                           # If true run simulation with noise
Nghost = 0                              # The number of ghost symbols inserted from each side


# amplification scheme
lumped = True                       # If true use lumped amplification
if not lumped:
    Nspans = 1
dz = 0.2                            # Z-step, [km] - initial step estimate
Nsps = np.ceil(La/dz)               # Required number of steps per span
dz = La/Nsps                        # Adjusting step slightly
L = La*Nspans                       # Total propagation distance
alphadB = 0.2                       # Power loss db/km

# Linear loss
if lumped:
    alpha = alphadB*np.log(10)/10
else:
    alpha = 0
G = np.exp(alpha*La)                # The power gain factor
gamma_eff = gamma                   # The effective gamma-coefficient in r.w.u.
if lumped:
    if G > 1:
        gamma_eff = gamma*(G-1)/(G*np.log(G))


# modulation constellation
mu = 0.1                            # Dimensionless power scaling factor (RRC)
M = 64                              # The order of M-QAM

# carrier characteristics
bet = 0.2                           # The roll-off factor

# normalized units
Tn = T0/(np.pi*(1+bet))             # [ps]
Zn = Tn**2/abs(beta2)               # [km]
Pn = 1/(gamma_eff*Zn)               # [W]

# Setting the time and (nonlinear) frequency arrays for each burst
Nos = 16                                # Oversampling factor (must be even)
Ns = (Nsc+2*Nghost)*Nos                 # The number of meaningful points
Nnft = int(4*2**(np.ceil(np.log2(Ns)))) # The number of points for NFT - round up to a power of 2
# Sometimes additional increase in #points is needed! 
Tnft = np.pi*Nos*Tn                     # The NFT base
dt = Tnft/Nnft                          # The time step in ps
Nb = 2*round(Tb/(2*dt))                 # The number of points in each burst
T1 = dt*(-Nnft/2)/Tn
T2 = dt*(Nnft/2-1)/Tn
_, xi_lims = nsev_inverse_xi_wrapper(Nnft,T1,T2,Nnft,display_c_msg=True)
xi = np.arange(-Ns/2, Ns/2-1)/Nos       # Array of upsampled nonlinear frequencies

# starting the loop

# Creating symbols and bursts
Nsc = int(Nspans*np.log2(M))        # Number of subcarriers
print('Generating bursts')
dataMod = np.zeros((Nbursts, Nsc), dtype=complex)   # Modulated message for each burst
uin = np.zeros((Nbursts, Ns), dtype=complex)        # Initial NFT spectrum for each burst
psi_xi = rrc_impulse(xi, 1, bet)                    # The xi-domain samples Cin=of the carriers
for ii in range(Nbursts):
    # Generate message
    message = np.random.randint(0, M, size=Nsc+2*Nghost)  # Integer message [1,M]
    print(f'lenght of message: {len(message)}')
    modem = QAMModem(M, gray_map=True)                      # Symbols sequence including ghost symbol
    c = modem.modulate(message)                        # Symbols sequence including ghost symbols
    dataMod[ii, :] = np.concatenate((c[0:Nsc//2], c[Nsc//2+2*Nghost+1:Nsc+2*Nghost]))  # Skiping ghost symbols in dataMod
    c[Nsc//2+1:Nsc//2+2*Nghost] = 0                        # Zeroing ghost symbols in the auxilliary array
    c = np.upsample(c, Nos)                                # Upsample the symbols
    # Perform the convolution to get the input sequence
    uin[ii, :] = mu*np.fft.ifft(np.fft.fft(psi_xi)*np.fft.fft(c))