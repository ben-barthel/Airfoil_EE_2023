clear all
close all
clc
%% Load Vorticity snapshots
[x,y,t,w,xwing,ywing] = load_vorticity();
%% Compute Temporal Wavelet Transform
[WTB,s,fv] = surface_WT(w,x,y,t,xwing,ywing);
%% Spatial Fourier transform w.r.t arc length
[WTF,ks] = surface_FFT(WTB,s);
%% save
% save('WT_boundary_FFT.mat','WTF','ks','t','fv'); 
% disp('Data Saved')