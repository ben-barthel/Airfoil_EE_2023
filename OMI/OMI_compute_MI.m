function [I_Pq,I_Pq_EE,I_PP] = OMI_compute_MI(P,q,start,T,cut_off,pdf_method)
%% Data Subset
stop = start + T;
dt = tau/(t(2)-t(1));
q = q(start+dt:stop+dt);
P = P(:,start:stop);
%% Precompute Mutual Info
I_Pq = zeros(100,1);
I_Pq_EE = zeros(100,1);
I_PP = zeros(100,100);
nbins = 16; % number of bins for monte carlo simulation
for j = 1:100
    if strcmp(pdf_method,'MC')
        I_Pq(j) = mutualInfo_MC(P(j,:),q,nbins);
        I_Pq_EE(j) = mutualInfo_MC_EE(P(j,:),q,nbins,cut_off);
    elseif strcmp(pdf_method,'KDE')
        I_Pq(j) = mutualInfo_KDE(P(j,:),q,16);
        I_Pq_EE(j) = mutualInfo_KDE_EE(P(j,:),q,24,cut_off);
    else
        disp('Please choose MC or KDE...')
    end
    for k = 1:100
        if strcmp(pdf_method,'MC')
            I_PP(j,k) = mutualInfo_MC(P(j,:),P(k,:),nbins);
        elseif strcmp(pdf_method,'KDE')
            I_PP(j,k) = mutualInfo_KDE(P(j,:),P(k,:),24);
        else
            disp('Please choose MC or KDE...')
        end
    end
end
disp('Pre-Processing Complete')
end