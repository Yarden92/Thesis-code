function u1= ssftm(u0,dt,dz,nz,beta2,gamma,alpha,D)

% This function solves the NSE for pulse propagation in an optical fiber using the split-step Fourier transfer method
% The distributed noise is added to the signal at each time step.
% The assumed form of eq:
% \partial A/\partial z = -(i/2)\beta_2 \partial^2 A/\partial t^2 + 
% i \gamma |A|^2 A -(\alpha/2) A+ N

% INPUT parameters:
%
% u0      = starting elctric field. Make it an even-numbered vector!
% dt      = time step
% dz      = propagation step-size
% nz      = number of steps to take, i.e., ztotal = dz*nz
% beta2   = beta2 coefficient
% gamma   = fibre nonlinearity coefficient
% alpha   = power loss coefficient (linear not dB!)
% 2D      = noise PSD, i.e., E[n(t1,z1)*n(t2,z2)] = 2D*delta(t2-t1,z2-z1)
% OUTPUT parameters:
%
% u1 = output electric field


nt = length(u0);
w = 2*pi*[(0:nt/2-1),(-nt/2:-1)]/(dt*nt);
halfstep = exp((1i*beta2*(w).^2/2)*dz/2-(alpha/2)*dz/2);
fullstep = exp(1i*beta2*(w).^2/2*dz-(alpha/2)*dz);
% Start with a linear half step
% Move to the fourier domain
ufft = fft(u0);
% Multiply by halfstep and go back
utime = ifft(halfstep.*ufft);
for iz = 1:nz-1
    %The nonlinear step
    utime = utime .* exp(1i*gamma*(abs(utime).^2)*dz);
    %Add noise
    noise = sqrt(D*dz/dt)*(randn(1,nt)+1i*randn(1,nt));
    utime = utime + noise;
    %Go full step
    % Move to the fourier domain
    ufft = fft(utime);
    % Multiply by full step and go back
    utime = ifft(fullstep.*ufft);
end
%Perform the remaining nonlinear step + noise + linear half step
utime = utime .* exp(1i*gamma*(abs(utime).^2)*dz);
noise = sqrt(D*dz/dt)*(randn(1,nt)+1i*randn(1,nt));
utime = utime + noise;
ufft = fft(utime);
u1 = ifft(halfstep.*ufft);
 