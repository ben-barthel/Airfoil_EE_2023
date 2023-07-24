function [ee_mode_norm,ee_mode_proj,vs_mode_norm,vs_mode_proj] = global_metrics(WT04,WT144,x,y,t,mask_wing)
%% Wavelet Scalar Metrics
dx = x(2)-x(1);
dy = y(2)-y(1);
[~,ind0] = min(abs(t - 911));
for j = 1:length(t)
    % EE Mode
    snap0 = (squeeze(WT04(:,:,ind0))');
    snap = (squeeze(WT04(:,:,j))');
    % norm
    W2dx = trapz(conj(snap).*mask_wing.*snap,1)*dx;
    W2dxdy = trapz(W2dx,2)*dy;
    ee_mode_norm(j) = abs(W2dxdy);
    % projection
    Wp2dx = trapz(conj(snap).*mask_wing.*snap0,1)*dx;
    Wp2dxdy = trapz(Wp2dx,2)*dy;
    ee_mode_proj(j) = abs(Wp2dxdy);
    clear snap snap0

    % VS Mode
    snap0 = (squeeze(WT144(:,:,ind0))');
    snap = abs(squeeze(WT144(:,:,j))');
    % norm
    W2dx = trapz(conj(snap).*mask_wing.*snap,1)*dx;
    W2dxdy = trapz(W2dx,2)*dy;
    vs_mode_norm(j) = abs(W2dxdy);
    % projection
    Wp2dx = trapz(conj(snap).*mask_wing.*snap0,1)*dx;
    Wp2dxdy = trapz(Wp2dx,2)*dy;
    vs_mode_proj(j) = abs(Wp2dxdy);

end
end