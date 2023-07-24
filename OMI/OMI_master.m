clear all
close all
clc

%% Inputs
pdf_method = 'MC'; % 'KDE' or 'MC'
cut_off = 2; % cut off for EE conditioned mutual information (std's)
start = 2000; % first data point to use
T = 90000; % number of points to use, 0 uses all points
filterP = 0; % 0 for raw pressure, 0 for filtered pressure
% Active learning inputs
Nsens = 6; % number of Sensors
window = 1; % window to smooth acquisition function - 1 uses no smoothing
% tau
tau = 0; % lead time for forecasting

%% Load Data
[t,P,q] = load_airfoil_data(filterP);
s = linspace(0,0.99,100);
%% Precompute Mutual Info
[I_Pq,I_Pq_EE,I_PP] = OMI_compute_MI(P,q,start,T,cut_off,pdf_method);
%% Active Learning
[OPT_IND] = OMI_search(I_pq,I_pq_EE,I_PP,s);
%% Plot
OMI_plot(OPT_IND,s);
