function [WTF,ks] = surface_FFT(WTB,s)
%% Spatial Fourier transform w.r.t arc length - dimensions are f,t,k
KS = 1/(s(2)-s(1));  % Sampling wavenumber
L = length(s);       % Length of signal
ks = KS*(0:(L/2))/L; % Wavenumber vector
% FFT 
PF1 = fft(WTB,[],2);
P2 = abs(PF1/L);
P1 = P2(:,1:L/2+1,:);
P1(:,2:end-1,:) = 2*P1(:,2:end-1,:);
WTF = permute(P1,[1,3,2]);

% for jf = 1:length(fv)
%     disp(['Starting Spatial Fourier Transforms for f = ',num2str(fv(jf))])
%     tic
%     for j = 1:length(t)
%         Pf1 = squeeze(WTB(jf,:,j));
%         PF1 = fft(Pf1);
%         P2 = abs(PF1/L);
%         P1 = P2(1:L/2+1);
%         P1(2:end-1) = 2*P1(2:end-1);
%         WTFs(jf,j,:) = smooth(P1,10);
%         WTF(jf,j,:) = P1;
%     end
%     toc
% end




end