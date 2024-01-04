%% NFDM - b-modulation simulator (Vanilla version)
%Input spectrum: \mu*\sum_k=1^Nsc ck*\psi_k(xi)
%Subcarrier shapes: RRC
%QAM modulation is assumed.
%Assumes half way precompensation 
%Propagates the field and plots the results
%Use it for preliminary estimates before b-miner
%% 
close all
clear
clc
addpath(genpath('/home/yarcoh/projects/thesis-code4/lib/FNFT-linux/matlab'));
output_dir = 'outputs';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% General system parameters
Nspans=12;                      % The number of spans
La = 80;                        % Transmission span [km]
beta2=-21.0;                    % GVD coefficient [ps^2/km]
gamma=1.27;                     % Nonlinear coefficient in [1/km*W]
eta=2;                          % SE penalty. eta=(T0+TG)/T0
W=0.05;                         % Total bandwidth, estimated [Thz]
TG=1.5*pi*W*abs(beta2)*La*Nspans; % The required GB [ps]
Nsc=round(W*TG/(eta-1));        % The number for subcarriers
Nsc=2*(round(Nsc/2));           % Round it up to the next even number
Nsc=round(2^ceil(log2(Nsc)));   % Round it up to the next power of 2
T0=Nsc/W;                       % Useful symbol duration, normalization time [ps]
%Tb=T0+TG;                       % Burst width [ps]
Tb=T0*eta;                     % Burst width
no_ss=false;                     % If true runs B2B run - no split step
noise=true;                    % If true run simulation with noise
%% Amplification scheme
lumped=false; %If true use lumped amplification
dz=0.2; %Z-step, [km] - initial step estimate
Nsps=ceil(La/dz); %Required number of steps per span
dz=La/Nsps; %Adjusting step slightly
L=La*Nspans; %Total propagation distance
%% Let's make some noise!
c_sol = 3e8;            % Speed of light [m/s]
h = 6.626e-34;      % Planck constant [J*s]
lambda0 = 1.55e-6;  % Carrier wavelength [m]
nu0 = c_sol/lambda0;    % Carrier frequency [Hz]
alphadB=0.2; % Power loss [db/km]
alpha=alphadB*log(10)/10; % Linear loss [1/km]

Kt=1.13; % Photon occupancy factor
D=0.5*h*nu0*1e12*Kt*alpha; % Noise PSD per quadrature [pJ/km]
alpha=0; % The loss is fully compensated inline

gamma_eff=gamma;

%% Modulation constellation
mu=0.15;                % Dimensionless power scaling factor (RRC)
M=16;                  % The order of M-QAM
%% Carrier charactersistics
bet=0.2; %The roll-off factor
%% Normalised units
Tn=T0/(pi*(1+bet)); % [ps] 
Zn=Tn^2/abs(beta2); % [km]
Pn=1/(gamma_eff*Zn); % [W]
%% Setting the time and (nonlinear) frequency arrays for each burst
Nos=16; % Oversampling factor (must be even)
Ns=Nsc*Nos; % The number of meaningful points
Nnft=4*2^(ceil(log2(Ns))); % The number of points for NFT - round up to a power of 2 
% Additional increase x4 in #points is needed!
Tnft=pi*Nos*Tn; % The NFT base
dt=Tnft/Nnft; % The time step in ps
Nb=2*round(Tb/(2*dt)); % The number of points in each burst
tau=dt*[-Nnft/2+1,Nnft/2]/Tn;
[XI, xi1] = mex_fnft_nsev_inverse_XI(Nnft, tau, Nnft);
% Array of upsampled nonlinear frequencies
xi=(-Ns/2:Ns/2-1)/Nos;
%% Creating symbols and bursts
tic
psi_xi=rrc_impulse(xi,1,bet); % The xi-domain samples Cin=of the carriers

%Generate message
message=randi([0 M-1],1,Nsc); %Integer message [1,M]
save_array_to_file(message, output_dir, 'message');
c_in=qammod(message,M,'gray'); % Symbols sequence including ghost symbols
save_array_to_file(c_in, output_dir, 'c_in');
dataMod_in=c_in; % Skiping ghost symbols in dataMod_in
c_in1=upsample(c_in,Nos); % Upsample the symbols
save_array_to_file(c_in1, output_dir, 'c_in1');
% Perform the convolution to get the input sequence
uin=mu*ifft(fft(psi_xi).*fft(c_in1));
save_array_to_file(uin, output_dir, 'uin');

fprintf('Time elapsed to generate bursts: %5.2f s\n',toc);
%% INFT @Tx
tic
Nb=min(Nb,Nnft); %If the estimated burst size is less than NFT base - truncate the latter
t=(-Nb/2+1:Nb/2)*dt; % Time array in r.w.u. 
q=zeros(1,Nb); %The dimensionles field in optical domain obtained by concatenation

%Applying exponential/linear scaling
%uin=mu*exp(-xi.^2/(2*20^2));
bin=sqrt(1-exp(-abs(uin).^2)).*exp(1i*angle(uin));
% Pre-compensation 
bin_pre=bin.*exp(-1i*xi.^2*(L/Zn)); 
save_array_to_file(bin_pre, output_dir, 'b_in')
% Pad with zeros symmetrically
bin_padded=zeros(1,Nnft);
bin_padded(Nnft/2-Ns/2:Nnft/2+Ns/2-1)=bin_pre;
x = bin_padded;
save_array_to_file(x, output_dir, 'x');
save_array_to_file(bin_padded, output_dir, 'b_in_padded')
qin=mex_fnft_nsev_inverse(bin_padded, XI, [], [], Nnft, tau, 1,'cstype_b_of_xi');
% Keeping only the central part of length N_b
save_array_to_file(qin, output_dir, 'q_in');
qb=qin((Nnft-Nb)/2+1:(Nnft+Nb)/2);
left=1;
right=Nb;
% Updating different parts of the main array
q(left:right)=qb;

q=q*sqrt(Pn); % Converting to the r.w.u.
save_array_to_file(q, output_dir, 'q_p');
fprintf('Time elapsed for INFT @Tx: %5.2f s\n',toc);
%% Split-step
%Average input optical power per burst [W]
Pave=dt*sum(abs(qb*sqrt(Pn)).^2)/Tb;
fprintf('The average power in dbm = %5.2f\n',30+10*log10(Pave));
%Calling the split-step routine
qz=q;
tic
if ~no_ss
    if lumped % TBD!
        % The loop is over fiber spans
        %qz=ssftm_niko_loss(q,dt,dz/Ln,Nspans,Nsps,-1,gamma_eff,alpha*Ln,nase*noise);
    else
        qz=ssftm_v2_prints(qz,dt,dz,Nsps*Nspans,beta2,gamma,D*noise, Tb);
        %w = 2*pi*[(0:Nb/2-1),(-Nb/2:-1)]/(dt*Nb);
        %fullstep = exp(1i*beta2*(w).^2/2*L);
        %ufft = fft(qz);
        %qz = ifft(fullstep.*ufft);
    end
end
save_array_to_file(qz, output_dir, 'qz');
fprintf('Time elapsed for split step: %5.2f s\n',toc);

I0=abs(q.^2)/1e-3;
I0=10*log10(I0);
I1=abs(qz.^2)/1e-3;
I1=10*log10(I1);
figure
plot(t,I0);
hold on
plot(t,I1);
line([(1/2)*Tb  (1/2)*Tb], [min(I1) max(I0)],'Color','red','LineStyle','--');

xlabel('Time, ps');
% ylabel('Power, mW');
ylabel('Power, dBm');
legend('Initial optical field','noised','Location','SouthWest')
xlim([-1,1]*Tb)
saveas(gcf,[output_dir,'/ssf_compare_both.png'])
%% Forward NFT @RX
tic
fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
qz=qz/sqrt(Pn); % Rescale the pulse to the soliton units

% Selecting current burst only
left=1;
right=Nb;
qb=qz(left:right);
%Pad with zeros up to size N_NFT
q_padded=zeros(1,Nnft);
if Nnft>Nb
    q_padded(Nnft/2-Nb/2+1:Nnft/2+Nb/2)=qb;
else % Just take the central part ot the burst
    q_padded=qb(Nb/2-Nnft/2+1:Nb/2+Nnft/2);    
end
save_array_to_file(q_padded, output_dir, 'q_padded');
%Forward NFT
contspec= mex_fnft_nsev(q_padded, tau, XI, +1, 'M', Nnft,'skip_bs','cstype_ab');
bout_padded=contspec(Nnft+1:2*Nnft); %b-coefficient is the second part
save_array_to_file(bout_padded, output_dir, 'b_out_padded');
% Select the central N_s = N_{sc} N_{os} samples
bout=bout_padded(Nnft/2-Ns/2:Nnft/2+Ns/2-1);
save_array_to_file(bout, output_dir, 'b_out');
% Postcompensation
if no_ss
    bout_post=bout.*exp(1i*xi.^2*(L/Zn));
else
    bout_post=bout.*exp(-1i*xi.^2*(L/Zn)); 
end
%Reversing the scaling
uout=sqrt(-log(1-abs(bout_post).^2)).*exp(1i*angle(bout_post));
save_array_to_file(uout, output_dir, 'u1_out');
figure();
plot(xi,real(uin),xi,real(uout))
xlabel('Nonlinear frequency, \xi');
ylabel('Re u(\xi)');
legend('Initial spectrum','Final spectrum','Location','SouthWest')
title(strcat('Burst #1'))


fprintf('Time elapsed for Forward NFT @Rx: %5.2f s\n',toc);
%% Demodulation and data processing (OFDM)
%uout = uin; % Short circuit for debugging (de)modulator

c_out1=ifft(fft(psi_xi).*fft(uout))/(mu*Nos); %Matched filtering
save_array_to_file(c_out1, output_dir, 'c_out1');
c_out=downsample(c_out1,Nos); %Downsampling
save_array_to_file(c_out, output_dir, 'c_out');
dataMod_Out=c_out; % Skiping ghost symbols in dataMod_in;

% plot stem bits
fig = figure();
stem(real(c_out));
xlabel('Bit index');
ylabel('Bit value');
title('Bits');
saveas(fig,[output_dir,'/bits.png'])

%% Constellation diagramm
sPlotFig =scatterplot(reshape(dataMod_in,[1,numel(dataMod_in)]),1,0,'w*');
%sPlotFig =scatterplot(positions,1,0,'k*');
hold on
%Updating the constellations
scatterplot(dataMod_Out,1,0,'go',sPlotFig);
set(sPlotFig, 'NumberTitle', 'off','Name', sprintf('%s','Constellation diagram'));
title('Constellation diagram');
xlabel('I')
ylabel('Q')
saveas(sPlotFig,[output_dir,'/constellations_v2.png'])
%figure
%plot(xi,angle(uout./uin))
%xlim([-80,80])

message_out = qamdemod(dataMod_Out,M,'gray');
binary_input = de2bi(message);
binary_output = de2bi(message_out);
% ber calculation
[number,ratio] = biterr(binary_input,binary_output);
fprintf('BER = %d/%d = %f\n',number,numel(binary_input),ratio);


function u1 = ssftm_v2_prints(u0,dt,dz,nz,beta2,gamma,D, Tb)  
    nt = length(u0);
    w = 2*pi*[(0:nt/2-1),(-nt/2:-1)]/(dt*nt);
    save_array_to_file(w, 'outputs', 'w');
    halfstep = exp((1i*beta2*(w).^2/2)*dz/2);
    fullstep = exp(1i*beta2*(w).^2/2*dz);
    % Start with a linear half step
    % Move to the fourier domain
    ufft = fft(u0);
    % Multiply by halfstep and go back
    utime = ifft(halfstep.*ufft);
    fprintf('looping %d times\n',nz-1');
    fprintf('noise amplitude = %.2e\n',sqrt(D*dz/dt));
    for iz = 1:nz-1
        %The nonlinear step
        utime = utime .* exp(1i*gamma*(abs(utime).^2)*dz);
        %Add noise
        noise = sqrt(D*dz/dt)*(randn(1,nt)+1i*randn(1,nt));
        
        if mod(iz, 500) == 0
            noise_power = 10*log10(dt*sum(abs(noise).^2)/Tb/1e-3);
            signal_power = 10*log10(dt*sum(abs(utime).^2)/Tb/1e-3);
            snr_db = signal_power - noise_power;
            fprintf('%04d: Noise power = %.2f, signal power = %.2f, SNR = %.2f dB\n',iz, noise_power,signal_power,snr_db);
        end
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

end

function Psi=rrc_impulse(t,Ts,bet)
    temp=bet*cos((1+bet)*pi*t/Ts);
    temp=temp+(pi*(1-bet)/4)*sinc((1-bet)*t/Ts);
    temp=temp./(1-(4*bet*t/Ts).^2);
    Psi=temp*4/(pi*sqrt(Ts));
    poles=isinf(temp);
    Psi(poles)= (bet/sqrt(Ts))*(-(2/pi)*cos(pi*(1+bet)/(4*bet))+sin(pi*(1+bet)/(4*bet)));
end


function save_array_to_file(array, dir, filename)
    path = [dir, '/', filename, '.mat']
    save(path, 'array');

end