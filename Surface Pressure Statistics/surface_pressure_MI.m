function [I_Pq,I_Pq_EE,I_Psq,I_Psq_EE,I_PP,cov_PP,I_PsPs,cov_PsPs] = surface_pressure_MI(tauv,method,nbins,cut_off)
%% Compute Mutual Information
% Load data
[t,P,q] = load_airfoil_data(0);
[~,Ps,~] = load_airfoil_data(1);
% Preallocate Variables
I_PP = zeros(100,100);cov_PP = zeros(100,100);
I_PsPs = zeros(100,100);cov_PsPs = zeros(100,100);
for jtau = 1:length(tauv)
    % set tau
    tau = tauv(jtau);
    % Set length of data to consider
    start = 1; stop = floor(length(t)/2);
    dt = floor(tau/(t(2)-t(1)));
    P = P(:,start:stop);
    Ps = Ps(:,start:stop);
    q = q(start+dt:stop+dt);
    t = t(start+dt:stop+dt);

    % Preallocate Variables
    I_Pq{jtau} = zeros(100,1); I_Pq{jtau} = zeros(100,1);
    I_Pq_EE{jtau} = zeros(100,1);I_Pq_EE{jtau} = zeros(100,1);
    rho_Pq{jtau} = zeros(100,1); varP{jtau} = zeros(100,1);
    for j = 1:100
        % Pressure - Drag Covariance
        % Raw Pressure
        R = cov(P(j,:),q);
        rho_Pq{jtau}(j) = R(1,2)/sqrt(R(1,1)*R(2,2));
        varPq{jtau}(j) = R(1,1);
        % Filtered PRessure
        Rs = cov(Ps(j,:),q);
        rho_Psq{jtau}(j) = Rs(1,2)/sqrt(Rs(1,1)*Rs(2,2));
        varPsq{jtau}(j) = Rs(1,1);

        % Pressure - Drag Mutual Information
        if strcmp(method,'MC')
            % Raw Pressure
            I_Pq{jtau}(j) = mutualInfo_MC(P(j,:),q,nbins);
            I_Pq_EE{jtau}(j) = mutualInfo_MC_EE(P(j,:),q,nbins,cut_off);
            % Filtered Pressure
            I_Psq{jtau}(j) = mutualInfo_MC(Ps(j,:),q,nbins);
            I_Psq_EE{jtau}(j) = mutualInfo_MC_EE(Ps(j,:),q,nbins,cut_off);
        elseif strcmp(method,'KDE')
            % Raw Pressure
            I_Pq{jtau}(j) = mutualInfo_KDE(P(j,:),q,nbins);
            I_Pq_EE{jtau}(j) = mutualInfo_KDE_EE(P(j,:),q,nbins,cut_off);
            % Filtered Pressure
            I_Psq{jtau}(j) = mutualInfo_KDE(Ps(j,:),q,nbins);
            I_Psq_EE{jtau}(j) = mutualInfo_KDE_EE(Ps(j,:),q,nbins,cut_off);
        else
            error('Please set method to MC or KDE')
        end
        % Intra-Pressure Covariance and Mutual Information
        if jtau == 1
            for k = 1:100
                I_PP(j,k) = mutualInfo_MC(P(j,:),P(k,:),nbins);
                I_PsPs(j,k) = mutualInfo_MC(Ps(j,:),Ps(k,:),nbins);
                R = cov(P(j,:),P(k,:));
                cov_PP(j,k) = R(1,2);
                Rs = cov(Ps(j,:),Ps(k,:));
                cov_PsPs(j,k) = Rs(1,2);
            end
        end
    end
end
disp('Mutual Information Computation Complete')


end