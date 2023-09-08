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
addpath(genpath('../../../../lib/FNFT-linux/matlab'));

%% General system parameters
Nbursts=1;                      % The number of bursts
Nspans=12;                      % The number of spans
La = 80;                        % Transmission span [km]
beta2=-21.0;                    % GVD coefficient [ps^2/km]
gamma=1.27;                     % Nonlinear coefficient in [1/km*W]
eta=2;                          % SE penalty. eta=(T0+TG)/T0
W=0.05;                         % Total bandwidth, estimated [Thz]
TG=1.5*pi*W*abs(beta2)*La*Nspans; % The required GB [ps]
Nsc=round(W*TG/(eta-1));        % The number for subcarriers
Nsc=2*(round(Nsc/2));           % Round it up to the next even number
T0=Nsc/W;                       % Useful symbol duration, normalization time [ps]
%Tb=T0+TG;                       % Burst width [ps]
Tb=T0*eta;                     % Burst width
no_ss=false;                     % If true runs B2B run - no split step
noise=true;                    % If true run simulation with noise
Nghost=2;                       % The number of ghost symbols inserted from each side
%% Amplification scheme
lumped=false; %If true use lumped amplification
dz=0.2; %Z-step, [km] - initial step estimate
Nsps=ceil(La/dz); %Required number of steps per span
dz=La/Nsps; %Adjusting step slightly
L=La*Nspans; %Total propagation distance
%% Let's make some noise!
c = 3e8;            % Speed of light [m/s]
h = 6.626e-34;      % Planck constant [J*s]
lambda0 = 1.55e-6;  % Carrier wavelength [m]
nu0 = c/lambda0;    % Carrier frequency [Hz]
alphadB=0.2; % Power loss [db/km]
alpha=alphadB*log(10)/10; % Linear loss [1/km]
if lumped
    G=exp(alpha*La); %T he power gain factor
    NFdB=5;        % EDFA noise figure, dB
    NF=10^(NFdB/10); % EDFA noise figure, linear
    F=(G*NF)/(2*(G-1)); % Amplifier inversion factor, Eq. 3.17 Mecozzi book
    NASE = h*nu0*(G-1)*F; % PSD of ASE, [J], Eq. 5.15 of Mecozzi book. 
else
    Kt=1.13; % Photon occupancy factor
    D=0.5*h*nu0*1e12*Kt*alpha; % Noise PSD per quadrature [pJ/km]
    alpha=0; % The loss is fully compensated inline
end
if lumped
    if G>1
        gamma_eff=gamma*(G-1)/(G*log(G)); %The effective gamma-coefficient in r.w.u.
    end
else
    gamma_eff=gamma;
end
%% Modulation constellation
mu=0.15;                % Dimensionless power scaling factor (RRC)
M=32;                  % The order of M-QAM
%% Carrier charactersistics
bet=0.2; %The roll-off factor
%% Normalised units
Tn=T0/(pi*(1+bet)); % [ps] 
Zn=Tn^2/abs(beta2); % [km]
Pn=1/(gamma_eff*Zn); % [W]
%% Setting the time and (nonlinear) frequency arrays for each burst
Nos=16; % Oversampling factor (must be even)
Ns=(Nsc+2*Nghost)*Nos; % The number of meaningful points
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
disp('Generating bursts')
dataMod=zeros(Nbursts,Nsc); %Modulated message for each burst
uin=zeros(Nbursts,Ns); % Initial NFT spectrum for each burst
psi_xi=rrc_impulse(xi,1,bet); % The xi-domain samples Cin=of the carriers
for ii=1:Nbursts
     %Generate message
    message=randi([0 M-1],1,Nsc+2*Nghost); %Integer message [1,M]
    c=qammod(message,M,'gray'); % Symbols sequence including ghost symbols
    dataMod(ii,:)=[c(1:Nsc/2), c(Nsc/2+2*Nghost+1:Nsc+2*Nghost)]; % Skiping ghost symbols in dataMod
    c(Nsc/2+1:Nsc/2+2*Nghost)=0; % Zeroing ghost symbols in the auxilliary array 
    c=upsample(c,Nos); % Upsample the symbols
    % Perform the convolution to get the input sequence
    uin(ii,:)=mu*ifft(fft(psi_xi).*fft(c));
end
disp('Done')
toc
%% INFT @Tx
tic
Nb=min(Nb,Nnft); %If the estimated burst size is less than NFT base - truncate the latter
t=(-Nb*Nbursts/2+1:Nb*Nbursts/2)*dt; % Time array in r.w.u. 
q=zeros(1,Nbursts*Nb); %The dimensionles field in optical domain obtained by concatenation
for ii=1:Nbursts  %The outer loop is over bursts
    %Applying exponential/linear scaling
    %uin=mu*exp(-xi.^2/(2*20^2));
    bin=sqrt(1-exp(-abs(uin(ii,:)).^2)).*exp(1i*angle(uin(ii,:)));
    % Pre-compensation 
    bin_pre=bin.*exp(-1i*xi.^2*(L/Zn)); 
    % Pad with zeros symmetrically
    bin_padded=zeros(1,Nnft);
    bin_padded(Nnft/2-Ns/2:Nnft/2+Ns/2-1)=bin_pre;
    qin=mex_fnft_nsev_inverse(complex(bin_padded), XI, [], [], Nnft, tau, 1,'cstype_b_of_xi');
    % Keeping only the central part of length N_b
    qb=qin((Nnft-Nb)/2+1:(Nnft+Nb)/2);
    left=1+Nb*(ii-1);
    right=left+Nb-1;
    % Updating different parts of the main array
    q(left:right)=qb;
end
q=q*sqrt(Pn); % Converting to the r.w.u.
toc
disp('(for INFT @Tx)');
%% Split-step
%Average input optical power per burst [W]
Pave=dt*sum(abs(q((Nbursts*Nb-Nb)/2+1:(Nb*Nbursts+Nb)/2)).^2)/Tb;
fprintf('The average power in dbm = %5.2f\n',10*log10(Pave/1e-3));
%Calling the split-step routine
qz=q;
tic
if ~no_ss
    if lumped % TBD!
        % The loop is over fiber spans
        %qz=ssftm_niko_loss(q,dt,dz/Ln,Nspans,Nsps,-1,gamma_eff,alpha*Ln,nase*noise);
    else
        % qz_unnoised=ssftm(qz,dt,dz,Nsps*Nspans/dz,beta2,gamma,0,false);
        qz_noised=ssftm(qz,dt,dz,Nsps*Nspans/dz,beta2,gamma,D*noise,true);
        %w = 2*pi*[(0:Nb/2-1),(-Nb/2:-1)]/(dt*Nb);
        %fullstep = exp(1i*beta2*(w).^2/2*L);
        %ufft = fft(qz);
        %qz = ifft(fullstep.*ufft);
    end
end
toc
disp('-----------------------------------')
disp('(for split-step)');
I0=abs(q.^2)/1e-3;
%I0=10*log10(I0);
I1=abs(qz_noised.^2)/1e-3;
% I2=abs(qz_noised.^2)/1e-3;
%I1=10*log10(I1);
figure
plot(t,I0);
hold on
plot(t,I1);
% hold on
% plot(t,I2);
for ii=0:Nbursts
    line([(ii-1/2-(Nbursts-1)/2)*Tb  (ii-1/2-(Nbursts-1)/2)*Tb], ...
        [min(I1) max(I0)],'Color','red','LineStyle','--');
end
xlabel('Time, ps');
ylabel('Power, mW');
legend('Initial optical field','noised','Location','SouthWest')
xlim([-1,1]*Tb*Nbursts)
saveas(gcf,'ssf_compare_both.png')
%% Forward NFT @RX
tic
fig = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
uout=zeros(Nbursts,Ns); % Final NFT spectrum for each burst
qz=qz/sqrt(Pn); % Rescale the pulse to the soliton units
for ii=1:Nbursts
    % Selecting current burst only
    left=1+Nb*(ii-1);
    right=left+Nb-1;
    qb=qz(left:right);
    %Pad with zeros up to size N_NFT
    q_padded=zeros(1,Nnft);
    if Nnft>Nb
        q_padded(Nnft/2-Nb/2+1:Nnft/2+Nb/2)=qb;
    else % Just take the central part ot the burst
        q_padded=qb(Nb/2-Nnft/2+1:Nb/2+Nnft/2);    
    end
    %Forward NFT
    contspec= mex_fnft_nsev(q_padded, tau, XI, +1, 'M', Nnft,'skip_bs','cstype_ab');
    bout_padded=contspec(Nnft+1:2*Nnft); %b-coefficient is the second part
    % Select the central N_s = N_{sc} N_{os} samples
    bout=bout_padded(Nnft/2-Ns/2:Nnft/2+Ns/2-1);
    % Postcompensation
    if no_ss
        bout_post=bout.*exp(1i*xi.^2*(L/Zn));
    else
        bout_post=bout.*exp(-1i*xi.^2*(L/Zn)); 
    end
    %Reversing the scaling
    uout(ii,:)=sqrt(-log(1-abs(bout_post).^2)).*exp(1i*angle(bout_post));
    subplot(1,Nbursts,ii);
    plot(xi,real(uin(ii,:)),xi,real(uout(ii,:)))
    xlabel('Nonlinear frequency, \xi');
    ylabel('Re u(\xi)');
    legend('Initial spectrum','Final spectrum','Location','SouthWest')
    title(strcat('Burst #',num2str(ii)))
end
toc
disp('(for forward NFT)');
%% Demodulation and data processing (OFDM)
dataModOut=zeros(Nbursts,Nsc); %Received constellation points
%uout = uin; % Short circuit for debugging (de)modulator
for ii=1:Nbursts
    c=ifft(fft(psi_xi).*fft(uout(ii,:)))/(mu*Nos); %Matched filtering
    c=downsample(c,Nos); %Downsampling
    dataModOut(ii,:)=[c(1:Nsc/2), c(Nsc/2+2*Nghost+1:Nsc+2*Nghost)]; % Skiping ghost symbols in dataMod;
end
%% Constellation diagramm
sPlotFig =scatterplot(reshape(dataMod,[1,numel(dataMod)]),1,0,'w*');
%sPlotFig =scatterplot(positions,1,0,'k*');
hold on
%Updating the constellations
scatterplot(dataModOut,1,0,'go',sPlotFig);
set(sPlotFig, 'NumberTitle', 'off','Name', sprintf('%s','Constellation diagram'));
title('Constellation diagram');
xlabel('I')
ylabel('Q')
%figure
%plot(xi,angle(uout./uin))
%xlim([-80,80])