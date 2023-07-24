function [P_hat,Ps_hat,f_fft] = surface_pressure_FFT(t,P,Ps,q,smooth_window)
%% FFT
clear mFA
[~,it1] = min(abs(t-100));
[~,it2] = min(abs(t-720));
tf = t(it1:it2-1);
Fs = 1/0.01;            % Sampling frequency
T_samp = 0.01;               % Sampling period
L = length(tf);       % Length of signal
f = Fs*(0:(L/2))/L;
P_hat = zeros(100,length(f));
Ps_hat = zeros(100,length(f));
ind = [0:1:100];
for j = 1:length(ind)
    if ind(j) == 0
        Pf1 = q(it1:it2-1);
        PF1 = fft(Pf1);
        P2 = abs(PF1/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);

    else
        % Raw Pressure
        Pf1 = P(ind(j),it1:it2-1);
        PF1 = fft(Pf1);
        P2 = abs(PF1/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [~,indFpeak] = min(abs(f-0.4));
        mFA(j-1) = P1(indFpeak);
        P_hat(ind(j),:) = smooth(P1,smooth_window);
        clear P1 P2 Pf1
        % Filtered Pressure
        Pf1 = Ps(ind(j),it1:it2-1);
        PF1 = fft(Pf1);
        P2 = abs(PF1/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        [~,indFpeak] = min(abs(f-0.4));
        mFA(j-1) = P1(indFpeak);
        Ps_hat(ind(j),:) = smooth(P1,smooth_window);


    end
    clear P1 P2
end
f_fft = f;
disp('Fourier Transform Computation Complete')
end