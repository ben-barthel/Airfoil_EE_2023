% Compute Wavelet Data Base

clear all
close all
clc
name = 'amor'
Nt = 50000;
% observable_tag = 35;
[t,P,q] = load_airfoil_data(0);
[~,P_smooth,~] = load_airfoil_data(1);
dt = t(2) - t(1);

%%



for j = 1:100
    disp(['j = ',num2str(j)])


    Fs = 1/(t(2)-t(1));
    data_in = P_smooth(j,:); data_in  = data_in - mean(data_in);
    tt = linspace(min(t),max(t),Nt);
   
    ff = linspace(0,2,81);
    [TT,FF] = meshgrid(tt,ff);
    % Wavelet Transform
    [wt,fwt] = cwt(data_in,name,Fs);
    WT = interp2(t,fwt,wt,TT,FF);

    [~,ind] = min(abs(ff-0.4) );
    WT04 = abs(WT(ind,:));
    dWT04 = gradient(WT04);
    d2WT04 = gradient(dWT04);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Drag Coefficent
    q0 = interp1(t,q,tt);
    % Pressure data
    Ps0 = interp1(t,P_smooth(j,:),tt);
    P0 = interp1(t,P(j,:),tt);


    %Wavelet Coefficents
%     MI0(j) = mutualInfo_MC(WT04,q0,34);
%     MI1(j) = mutualInfo_MC(dWT04,a0,34);
%     MIEE0(j) = mutualInfo_MC_EE(WT04,q0,34,cut_off);
%     MIEE1(j) = mutualInfo_MC_EE(dWT04,a0,34,cut_off);

    wavelet(j,:,:) = WT;
   
    gamma.gm = WT04;
    gamma.dgm = dWT04;
    gamma.d2gm = d2WT04;
    gamma.q = q0;
    gamma.t = tt;
    gamma.P = P0;
    gamma.P_smooth = Ps0;

    waveletDB{j} = gamma;
    clear gamma
end

save('waveletDB.mat','waveletDB')
disp('Data Saved')