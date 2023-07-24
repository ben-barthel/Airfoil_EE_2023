clear all
close all
clc
%% Inputs
smooth_window = 100; % smoothing window for FFT plot
sensor_tag = 35; % Sensor number for wavelet transform plot: 1-100
tauv = [0,3,7]; % Lead times
cut_off = 2; % cut off for Extrem Event Mutual Information (in units of standard deviations)
method = 'MC'; % Choose "MC" for Monte Carlo or "KDE" for Kernel Density Estimation
nbins = 34; % bins for Monte carlo or KDE estimation
%% Load In Data
[t,Ps,q] = load_airfoil_data(1);
[~,P,~] = load_airfoil_data(0);
s = linspace(0,0.99,100); % arc length variable
%% Mutual Information and Covariance
[I_Pq,I_Pq_EE,I_Psq,I_Psq_EE,I_PP,cov_PP,I_PsPs,cov_PsPs] = surface_pressure_MI(tauv,method,nbins,cut_off);
%% FFT
[P_hat,Ps_hat,f_fft] = surface_pressure_FFT(t,P,Ps,q,smooth_window);
%% Wavelet Transform
[WT,WT04,t_wt,f_wt] = surface_pressure_WT(t,Ps,sensor_tag);
%% Example Data Plots
figure(1);close;figure(1);
subplot(3,1,1)
plot(t,q,'k','LineWidth',1)
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlabel('$t$','Interpreter','latex');
ylabel('$q(t)$','Interpreter','latex');
xlim([600,900])
subplot(3,1,2)
plot(t,P(25,:),'k','LineWidth',1)
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlabel('$t$','Interpreter','latex');
ylabel('$P(t)$','Interpreter','latex');
xlim([600,900])
subplot(3,1,3)
plot(t,Ps(25,:),'k','LineWidth',1)
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlabel('$t$','Interpreter','latex');
ylabel('$\tilde{P}(t)$','Interpreter','latex');
xlim([600,900])

%% Intra-Pressure Mutual Information and Covariance Plots
map = 'jet';
figure(2);close;figure(2)
subplot(1,2,1)
contourf(s,s,I_PP,200,'linestyle','none')
xticks([0,0.25,0.5,0.75,1.0]);
yticks([0,0.25,0.5,0.75,1.0]);
xlabel('$s$','interpreter','latex');
ylabel('$s$','interpreter','latex');
set(gca,'fontsize',30,'fontname','times')
axis equal;
colormap(map);
colorbar
subplot(1,2,2)
contourf(s,s,I_PsPs,200,'linestyle','none')
xticks([0,0.25,0.5,0.75,1.0]);
yticks([0,0.25,0.5,0.75,1.0]);
xlabel('$s$','interpreter','latex');
ylabel('$s$','interpreter','latex');
set(gca,'fontsize',30,'fontname','times')
axis equal;
colormap(map);
colorbar

figure(3);close;figure(3)
subplot(1,2,1)
contourf(s,s,cov_PP,200,'linestyle','none')
xticks([0,0.25,0.5,0.75,1.0]);
yticks([0,0.25,0.5,0.75,1.0]);
xlabel('$s$','interpreter','latex');
ylabel('$s$','interpreter','latex');
set(gca,'fontsize',30,'fontname','times')
axis equal;
colormap(map); caxis([-1,1])
colorbar
subplot(1,2,2)
contourf(s,s,cov_PsPs,200,'linestyle','none')
xticks([0,0.25,0.5,0.75,1.0]);
yticks([0,0.25,0.5,0.75,1.0]);
xlabel('$s$','interpreter','latex');
ylabel('$s$','interpreter','latex');
set(gca,'fontsize',30,'fontname','times')
axis equal;
colormap(map); caxis([-1,1])
colorbar

%% Pressure-Drag Mutual Information Plots
plot_color ={'-ob','->r','-xg'};
lw = 2;
ms = 7; 
mi = 4;
figure(4);close;figure(4)
for jtau = 1:length(tauv)
    % Normalize Mutual Information Profiles for Plotting
    I_Pq_plot = I_Pq{jtau}./max(I_Pq{jtau});
    I_Pq_EE_plot = I_Pq_EE{jtau}./max(I_Pq_EE{jtau});
    I_Psq_plot = I_Psq{jtau}./max(I_Psq{jtau});
    I_Psq_EE_plot = I_Psq_EE{jtau}./max(I_Psq_EE{jtau});

    figure(4)
    subplot(2,2,1)
    plot(s,I_Pq_plot,plot_color{jtau},'linewidth',lw,'MarkerSize',ms,'MarkerIndices',[1:mi:length(s)]); hold on;
    xticks([0,0.25,0.5,0.75,1.0]);
    xlabel('$s$','interpreter','latex');
    ylabel('$I(P,q)$','interpreter','latex')
    set(gca,'fontsize',20,'fontname','times')
    grid on;  %axis equal;
    ylim([0,1.1])

    subplot(2,2,2)
    plot(s,I_Pq_EE_plot,plot_color{jtau},'linewidth',lw,'MarkerSize',ms,'MarkerIndices',[1:mi:length(s)]); hold on;
    xticks([0,0.25,0.5,0.75,1.0]);
    xlabel('$s$','interpreter','latex');
    ylabel('$I_{EE}(P,q)$','interpreter','latex')
    set(gca,'fontsize',20,'fontname','times')
    grid on;  %axis equal;
    ylim([0.5,1.1])

    subplot(2,2,3)
    plot(s,I_Psq_plot,plot_color{jtau},'linewidth',lw,'MarkerSize',ms,'MarkerIndices',[1:mi:length(s)]); hold on;
    xticks([0,0.25,0.5,0.75,1.0]);
    xlabel('$s$','interpreter','latex');
    ylabel('$I(\tilde{P},q)$','interpreter','latex')
    set(gca,'fontsize',20,'fontname','times')
    grid on;  %axis equal;
    ylim([0,1.1])

    subplot(2,2,4)
    plot(s,I_Psq_EE_plot,plot_color{jtau},'linewidth',lw,'MarkerSize',ms,'MarkerIndices',[1:mi:length(s)]); hold on;
    xticks([0,0.25,0.5,0.75,1.0]);
    xlabel('$s$','interpreter','latex');
    ylabel('$I_{EE}(\tilde{P},q)$','interpreter','latex')
    set(gca,'fontsize',20,'fontname','times')
    grid on;  %axis equal;
    ylim([0.5,1.1])



end
%% Fourier Transform Plots
f_fft = f_fft(1:2000);
P_hat = P_hat(:,1:2000);
Ps_hat = Ps_hat(:,1:2000);
ff = repmat(f_fft,[100,1]);

figure(5);close;figure(5)
subplot(1,2,1)
contourf(s,f_fft,Ps_hat',100,'LineStyle','none')
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlabel('$s$','Interpreter','latex');ylabel('$f$','Interpreter','latex');
% title('$\hat{P}_{f}(s,f)$','Interpreter','latex')
colorbar;colormap('jet');
set(gca,'ColorScale','log')
ylim([0.1,1.2]);

subplot(1,2,2)
contourf(s,f_fft,(ff.*Ps_hat)',100,'LineStyle','none')
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlabel('$s$','Interpreter','latex');ylabel('$f$','Interpreter','latex');
% title('$f\hat{P}_{f}(s,f)$','Interpreter','latex')
colorbar;colormap('jet');
set(gca,'ColorScale','log')
ylim([0.1,1.2]);


%% wavelet Transform Plots
figure(6);close;figure(6)
subplot(2,1,1)
contourf(t_wt,f_wt,abs(WT),'linestyle','none')
ylim([0.025,1])
xlim([700,900])
set(gca,'fontsize',30,'fontname','times','YColor','k')
ylabel('$f$','Interpreter','latex');
colormap('jet');

subplot(2,1,2)
plot(t_wt,WT04,'->r','linewidth',2,'MarkerIndices',[1:4:length(t_wt)]); hold on;
set(gca,'fontsize',30,'fontname','times','YColor','k')
xlim([700,900]);
grid on
xlabel('$t$','Interpreter','latex');
ylabel('$\gamma$','Interpreter','latex');
yyaxis right
plot(t,q,'-k','linewidth',2); hold on;
set(gca,'fontsize',30,'fontname','times','YColor','k')
ylabel('$q$','Interpreter','latex');