function [u1, pn] = ssftm(u0,dt,dz,nz,beta2,gamma,D,display)

% This function solves the NSE for pulse propagation in an optical fiber using the split-step Fourier transfer method
% The distributed noise is added to the signal at each time step.
% Soliton units are used throughout

% INPUT parameters:
%
% u0      = starting elctric field. Make it an even-numbered vector!
% dt      = time step
% dz      = propagation step-size
% nz      = number of steps to take, i.e., ztotal = dz*nz
% beta2   = dimensionless beta2=-1 or -2 
% gamma   = dimensionless fibre nonlinearity parameter gamma=1 or 2
% 2D      = noise PSD, i.e., E[n(t1,z1)*n(t2,z2)] = 2D*delta(t2-t1,z2-z1)
% display = if true - display the progress
% OUTPUT parameters:
%
% u1 = output electric field
% pn = total noise power added

global noise
u0=reshape(u0,[1, length(u0)]);
nt = length(u0);
w = 2*pi*[(0:nt/2-1),(-nt/2:-1)]/(dt*nt);

halfstep = 1i*beta2*(w).^2/2;
halfstep = exp(halfstep*dz/2);

%u1 = u0;
ufft = fft(u0);
pn = 0;
if display
    if(dz>=0)
        h = waitbar(0,'Propagating. Please wait...');
    else
        h = waitbar(0,'Back-propagating. Please wait...');
    end
end
for iz = 1:nz
    uhalf = ifft(halfstep.*ufft);
    noise = sqrt(D*dz/dt)*(randn(1,nt)+1i*randn(1,nt));
    %pn = pn + norm(noise)^2/nt;
    uhalf = uhalf + noise;  
    uv = uhalf .* exp(1i*gamma*(abs(uhalf).^2)*dz);
    ufft = halfstep.*fft(uv);
    if display 
        if(rem(iz,4)==0) 
            waitbar(iz / nz)
        end
    end
end
if display
    close(h)
end
u1 = gather(ifft(ufft));