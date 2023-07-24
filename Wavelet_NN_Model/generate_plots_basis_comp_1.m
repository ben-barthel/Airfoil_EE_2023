clear all
close all
clc
%% Inputs
lossTypeVec = {'MAE','OWMAE'};
inputDataVec = {'rawP','smoothP','wavelet'};

tau = 7;
Nepoch = 500;
Ntest = 10;
%% Load Data
[Q,PQ] = wavelet_NN_load_results(tau,Nepoch,Ntest,lossTypeVec,inputDataVec);
%% PDF Plots
plot_color = {'g','b','r'};
figure(1);close;figure(1)
subplot(1,2,1)
semilogy(PQ{1,1}(:,1),PQ{1,1}(:,3),'k','linewidth',2.5); hold on
subplot(1,2,2)
semilogy(PQ{2,1}(:,1),PQ{2,1}(:,3),'k','linewidth',2.5); hold on
for jd = 1:length(inputDataVec)
    inputData = inputDataVec{jd};
    figure(1)
    subplot(1,2,1)
    semilogy(PQ{1,jd}(:,1),PQ{1,jd}(:,2),plot_color{jd},'linewidth',2.5); hold on
    subplot(1,2,2)
    semilogy(PQ{2,jd}(:,1),PQ{2,jd}(:,2),plot_color{jd},'linewidth',2.5); hold on
end
figure(1)
for j = 1:length(lossTypeVec)
    subplot(1,2,j)
    if j == 1
        legend('True','Raw Pres.','Smooth Pres.','Wavelet','location','northeast')
    end
    ylim([2*10^-2,3.5])
    xlim([-4,5])
    xticks([-4:2:4])
    grid on
    xlabel('$q$','interpreter','latex')
    ylabel('$p_q$','interpreter','latex')
    BBplotSettings(25,0)
end



%% Time Series Plots
figure(2);close;figure(2)
figure(3);close;figure(3)
for jd = 1:3
    inputData = inputDataVec{jd};
    % Time Series Plot 1 - OWMAE
    figure(2)
    subplot(3,1,jd)
    plot(Q{1,1}(:,1),Q{1,1}(:,4),'k','linewidth',2); hold on
    plot(Q{2,jd}(:,1),Q{2,jd}(:,2),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(20,1);
    if jd == 3
        xlabel('$t$','interpreter','latex');
    end
    ylabel('$q$','interpreter','latex');
    xlim([830,1000]);
    ylim([-2,4])
    
    
    % Time Series Plot 2 - MAE
    figure(3)
    subplot(3,1,jd)
    plot(Q{1,1}(:,1),Q{1,1}(:,4),'k','linewidth',2); hold on
    plot(Q{1,jd}(:,1),Q{1,jd}(:,2),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(20,1);
    if jd == 3
        xlabel('$t$','interpreter','latex');
    end
    ylabel('$q$','interpreter','latex');
    xlim([830,1000]);
    ylim([-2,4])
end
