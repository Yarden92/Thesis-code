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
%% General system parameters
Nbursts=1;                      % The number of bursts
Nspans=12;                      % The number of spans
La = 12*80/Nspans;              % Transmission span [km]
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
no_ss=true;                     % If true runs B2B run - no split step
noise=false;                    % If true run simulation with noise
Nghost=2;                       % The number of ghost symbols inserted from each side
%% Amplification scheme
lumped=true; %If true use lumped amplification
if ~lumped
    Nspans=1;
end
dz=0.2; %Z-step, [km] - initial step estimate
Nsps=ceil(La/dz); %Required number of steps per span
dz=La/Nsps; %Adjusting step slightly
L=La*Nspans; %Total propagation distance
alphadB=0.2; %Power loss db/km
%Linear loss
if lumped
    alpha=alphadB*log(10)/10;
else
    alpha=0;
end
G=exp(alpha*La); %The power gain factor
gamma_eff=gamma; %The effective gamma-coefficient in r.w.u.
if lumped
    if G>1
        gamma_eff=gamma*(G-1)/(G*log(G));
    end
end
%% Modulation constellation
mu=0.1;                % Dimensionless power scaling factor (RRC)
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
% Sometimes additional increase in #points is needed!
Tnft=pi*Nos*Tn; % The NFT base
dt=Tnft/Nnft; % The time step in ps
Nb=2*round(Tb/(2*dt)); % The number of points in each burst
tau=dt*[-Nnft/2,Nnft/2-1]/Tn;
[XI, xi1] = mex_fnft_nsev_inverse_XI(Nnft, tau, Nnft);
xi=(-Ns/2:Ns/2-1)/Nos;
% Array of upsampled nonlinear frequencies
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
    bin=sqrt(1-exp(-abs(uin(ii,:)).^2)).*exp(1i*angle(uin(ii,:)));
    % Pre-compensation 
    bin=bin.*exp(-1i*xi.^2*(L/Zn)); 
    % Pad with zeros symmetrically
    bin_padded=zeros(1,Nnft);
    bin_padded(Nnft/2-Ns/2+1:Nnft/2+Ns/2)=bin;
    qin=mex_fnft_nsev_inverse(bin_padded, XI, [], [], Nnft, tau, 1,'cstype_b_of_xi');
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
%% TBD. Currently does essentially nothing
%Average input optical power per burst [W]
Pave=dt*sum(abs(q((Nbursts*Nb-Nb)/2+1:(Nb*Nbursts+Nb)/2)).^2)/Tb;
fprintf('The average power in dbm = %5.2f\n',10*log10(Pave/1e-3));
%Calling the split-step routine
tic
qz=q;
%% The next part TBD:
%if ~no_ss
%    if lumped
%        qz=ssftm_niko_loss(q,dt,dz/Ln,Nspans,Nsps,-1,gamma_eff,alpha*Ln,nase*noise);
%    else
%        ssftm_niko(qz,dt,dz/Ln,Nsps*Nspans,-1,1,D*noise,true,true);
%        qz=ssftm_niko(qz,dt,dz/Ln,Nsps*Nspans,-1,1,D*noise,false,false);
%    end
%end
toc
disp('(for split-step)');
I0=abs(q.^2)/1e-3;
%I0=10*log10(I0);
I1=abs(qz.^2)/1e-3;
%I1=10*log10(I1);
figure
plot(t,I0);
hold on
plot(t,I1);
for ii=0:Nbursts
    line([(ii-1/2-(Nbursts-1)/2)*Tb  (ii-1/2-(Nbursts-1)/2)*Tb], ...
        [min(I1) max(I0)],'Color','red','LineStyle','--');
end
xlabel('Time, ps');
ylabel('Power, mW');
legend('Initial optical field','Final optical field','Location','SouthWest')
xlim([-1,1]*Tb*Nbursts)
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
    bout=bout_padded(Nnft/2-Ns/2+1:Nnft/2+Ns/2);
    % Postcompensation
    if no_ss
        bout=bout.*exp(1i*xi.^2*(L/Zn));
    else
        bout=bout.*exp(-1i*xi.^2*(L/Zn)); 
    end
    %Reversing the scaling
    uout(ii,:)=sqrt(-log(1-abs(bout).^2)).*exp(1i*angle(bout));
    subplot(1,Nbursts,ii);
    plot(xi,abs(uin(ii,:)),xi,abs(uout(ii,:)))
    xlabel('Nonlinear frequency, \xi');
    ylabel('|u(\xi)|');
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
