function [t,P,q] = load_airfoil_data(smoothP)


% Nbins use 16 with KDE and 64 with MC
% number of bins in histogram or points in KDE, 0 mean automatic choice
%% Read in Data


start = 2000;
% Read Pressure
if smoothP
    P = readNPY('../P_smooth.npy');t = P(:,1)';P = P(:,2:end);
else
    P = readNPY('../P.npy');t = P(:,1)';P = P(:,2:end);
end
% Read Drag Coefficent
q = readNPY('q.npy'); tq = q(:,1); q = q(:,2);
% interpolate
q = interp1(tq,q,t,'spline');
data_length = length(t) - start -1;
% Reshape P to be in terms of arclength
% P12 = P(:,1:50);P34 = flip(P(:,51:end),2);

P12 = P(:,1:51);P34 = flip(P(:,52:end),2);


stop = start + data_length;
P = [P12,P34];P = P(start:stop,:)';
Pbar = repmat(mean(P,2),[1,size(P,2)]);
P = P-Pbar;
q = q(start:stop); 
t = t(start:stop);
clear P12 P34
end