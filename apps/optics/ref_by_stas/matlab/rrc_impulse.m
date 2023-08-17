%Returns impulse responce of the root raised cosine filter
% Inputs:
% t - array of time domain values
% Ts - intersymbol distance
% bet - roll-off factor 
% Output:
% Psi - discrete samples of the RRC spectrum
function Psi=rrc_impulse(t,Ts,bet)
    temp=bet*cos((1+bet)*pi*t/Ts);
    temp=temp+(pi*(1-bet)/4)*sinc((1-bet)*t/Ts);
    temp=temp./(1-(4*bet*t/Ts).^2);
    Psi=temp*4/(pi*sqrt(Ts));
    poles=isinf(temp);
    Psi(poles)= (bet/sqrt(Ts))*(-(2/pi)*cos(pi*(1+bet)/(4*bet))+sin(pi*(1+bet)/(4*bet)));
end
