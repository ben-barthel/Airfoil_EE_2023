
% Plot Spatial FFT of Wavelet Transformed Boundary Vorticity
% ks: spatial wavenumber w.r.t. arch length vector
% t: time vector
% fv: temporal frequency vector
% WTF: Fourier transform w.r.t. arc length of Wavelet transformed vorticity
% evaluated at airfoil surface

%% Load Data
clear all 
close all
clc
load('WT_boundary_FFT.mat')
WTFp = permute(WTF,[2,3,1]);
[KS,T,FV] = meshgrid(ks,t,fv);
[tq,P,q] = load_airfoil_data(0);
[~,Ps,~] = load_airfoil_data(1);
%% Plot Inputs
% Contour Values
contour_val= [6,7.5];
n_contours = length(contour_val);
% Transparancy Values
alpha_val = linspace(0.5,0.9,n_contours);
% Colors
colors = hot(1+n_contours);


%% Generate Isocontours
clear is p
figure(1);close;figure(1);
for j = 1:n_contours
    is{j} = isosurface(KS,T,FV,WTFp,contour_val(j));
    p{j} = patch(is{j});
    isonormals(KS,T,FV,WTFp,p{j}); hold on;
   
end
view(3);
camlight;
lighting gouraud;

%% Format Isocontours
figure(1)

for j = 1:n_contours
    set(p{j},'FaceColor',colors(j,:),'EdgeColor','none','FaceAlpha',alpha_val(j));
end
set(gca,"FontName",'times','FontSize',25);
grid on;
xlabel('$k_s$','Interpreter','latex');
ylabel('$t$','Interpreter','latex');
zlabel('$f$','Interpreter','latex');

xlim([0,30])
zlim([0.1,0.8])
ylim([870,910])
% ylim([905,998])
view(76.9230, 32.2545)

EE1 = 884;
EE2 = 902;
eem = q./max(q); eem = eem - min(eem); eem =eem*0.3 + 0.082; 
y = ones([1,30]);
figure(1)
line(gca,linspace(0,30,30),y*EE1,ones([1,30])*0.1,'Color','b','LineStyle','--','LineWidth',2.5);
line(gca,linspace(0,30,30),y*EE2,ones([1,30])*0.1,'Color','b','LineStyle','--','LineWidth',2.5);
line(gca,ones([1,30])*0,y*EE1,linspace(0,2,30),'Color','b','LineStyle','--','LineWidth',2.5);
line(gca,ones([1,30])*0,y*EE2,linspace(0,2,30),'Color','b','LineStyle','--','LineWidth',2.5);
plot3(30*ones(size(tq)),tq,eem,'k','linewidth',3);
patch( 30*[1 -1 -1 1] , EE1*[1 1 1 1], [1 1 -1 -1], [1 1 -1 -1], 'FaceColor', 'b','FaceAlpha',0.15)
patch( 30*[1 -1 -1 1] , EE2*[1 1 1 1], [1 1 -1 -1], [1 1 -1 -1], 'FaceColor', 'b','FaceAlpha',0.15)

patch( 30*[1 -1 -1 1] , 1000*[1 1 -1 -1], 0.4*[1 1 1 1], [1 1 -1 -1], 'FaceColor', 'b','FaceAlpha',0.1)
% quiver3(18,EE1,0.42,-90,0,-1,'k','linewidth',1,'AutoScaleFactor',0.05,'ShowArrowHead','on')
