import numpy as np


def rrc_impulse(t: np.ndarray, Ts: float, bet: float) -> np.ndarray:
    # Returns impulse responce of the root raised cosine filter
    # Inputs:
    # t - array of time domain values
    # Ts - intersymbol distance
    # bet - roll-off factor

    # Output:
    # Psi - discrete samples of the RRC spectrum
    
    temp = bet*np.cos((1+bet)*np.pi*t/Ts)
    temp = temp + (np.pi*(1-bet)/4)*np.sinc((1-bet)*t/Ts)
    temp = temp/(1-(4*bet*t/Ts)**2)
    Psi = temp*4/(np.pi*np.sqrt(Ts))
    poles = np.isinf(temp)
    Psi[poles] = (bet/np.sqrt(Ts))*(-(2/np.pi)*np.cos(np.pi*(1+bet)/(4*bet))+np.sin(np.pi*(1+bet)/(4*bet)))
    return Psi