function [WTB,s,fv] = surface_WT(w,x,y,t,xwing,ywing)
%% Temporal Wavelet transform - dimensions are f,x,y,t
ny = length(y);
nx = length(x);
nt = length(t);
dt = t(2)-t(1);
dx = x(2)-x(1);
dy = y(2)-y(1);
[mask_wing] = airfoil_mask(x,y,xwing,ywing);
% frquencies to analyze
fv = [0.1:0.1:1.3,1.44];
% Only perform WT near the airfoil
iy1 = 175;iy2 = 230;
y_inds = [iy1:1:iy2];
nyp = length(y_inds);
nxp = nx/2;
xs = x(1:nx/2);
ys = y(iy1:iy2);
mask_wing_i = mask_wing(1:nxp,iy1:iy2);
nf = 81;
Fs = 1/(dt);
ff = linspace(0,2,nf);
% Make sure to include vortex shedding frequency
[~,ivs] = min(abs(ff-1.44));
ff(ivs) = 1.44;
[TT,FF] = meshgrid(t,ff);
name = 'amor';
disp('Starting Temporal Wavelet Transform...')
for jx = 1:nxp % x point
    for jy = 1:nyp % y point
        jyi = y_inds(jy);
        % check that point is not inside wing
        if mask_wing(jx,jy) == 0
            continue
        else
            % Wavelet Transform
            [wt,fwt] = cwt(squeeze(w(jx,jyi,:)),name,Fs);
            WTj = interp2(t,fwt,wt,TT,FF);
            % Extract Frequencies of interest
            for jf = 1:length(fv)
                [~,indf] = min(abs(ff-fv(jf)) );
                WT(jf,jx,jy,:) = (WTj(indf,:));
            end
        end
    end
end
%% Define Grid
% arc length
for j = 2:length(xwing)
    r(j) = sqrt((xwing(j)-xwing(j-1))^2 + (ywing(j)-ywing(j-1))^2);
end
sr = cumsum(r);sr = sr-min(sr);sr = sr/max(sr);
% uniformly spaced arc length grid for FFT
s = linspace(0,1,length(xwing));
%% Evaluate vorticity at boundary - dimensions are f,s,t
for jf = 1:length(fv)
    for j = 1:length(t)
        snap = squeeze(mask_wing_i.*squeeze(WT(jf,:,:,j)))';
        [wbj] = interp2(xs,ys,snap,xwing,ywing,'spline');
        wbj = interp1(sr,wbj,s);
        WTB(jf,:,j) = wbj;
    end
end
end