clear all
close all
clc
%% Inputs
saveWT = 1; % 1 for yes, 0 for no
%% Load Vorticity snapshots
[x,y,t,w,xwing,ywing] = load_vorticity();
%% Create mask
[mask_wing] = airfoil_mask(x,y,xwing,ywing);
%% Wavelet Transform
[WT04,WT144,ff] = global_WT(w,x,y,t);
%% Wavelet Scalar Metrics
[ee_mode_norm,ee_mode_proj,vs_mode_norm,vs_mode_proj] = global_metrics(WT04,WT144,x,y,t,mask_wing);
%% Save
if saveWT
    filename = ['vorticity_Wavelet_Transform_Nt',num2str(length(t)),'.mat'];
    save(filename,'t','ff','x','y','WT04','WT144','ee_mode_proj','ee_mode_norm','vs_mode_proj','vs_mode_norm','-v7.3');disp('wavelet data saved')
end
