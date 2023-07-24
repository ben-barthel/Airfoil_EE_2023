clear all
close all
clc
%% Load Data
[x,y,t,w,xwing,ywing] = load_vorticity();
[mask_wing] = airfoil_mask(x,y,xwing,ywing);

%% Full Solution - Farfield
figure(1);close;figure(1)
snap = squeeze(w(:,:,12))';
snap(abs(snap)<0.1) = 0;
contourf(x,y,snap,400,'LineStyle','none');
patch(xwing,ywing,'k')
BBplotSettings(20,0);
colormap('REDBLUE')
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
caxis([-50,50])
axis equal

%% Snap