clear all
close all
clc
%% Inputs
lossTypeVec = {'MAE','OWMAE'};
inputDataVec = {'rawP','smoothP','wavelet'};
tau = 7
%% Load Data

for jl = 1:2
    lossType = lossTypeVec{jl};
    for jd = 1:3
        inputData = inputDataVec{jd};
        % Test set Data: t, q_mean, q_std, q_true
        % Pdf data: val, pq_model, pq_true
        % Load
        name1 = ['../../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep201_Ntest10_',lossType,'_S_(5, 35, 65, 95)_out.npy'];
        name2 = ['../../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep201_Ntest10_',lossType,'_S_(5, 35, 65, 95)_pdf.npy'];
        Q{jl,jd}  = readNPY(name1);
        PQ{jl,jd} = readNPY(name2);

    end
end

%% Plots
% True Value
plot_color = {'g','b','r'};
figure(1);close;figure(1)
subplot(1,2,1)
semilogy(PQ{1,1}(:,1),PQ{1,1}(:,3),'k','linewidth',2.5); hold on
subplot(1,2,2)
semilogy(PQ{2,1}(:,1),PQ{2,1}(:,3),'k','linewidth',2.5); hold on


for jd = 1:3
    inputData = inputDataVec{jd};

    % PDF Plot
    figure(1)
    subplot(1,2,1)
    semilogy(PQ{1,jd}(:,1),PQ{1,jd}(:,2),plot_color{jd},'linewidth',2.5); hold on
    subplot(1,2,2)
    semilogy(PQ{2,jd}(:,1),PQ{2,jd}(:,2),plot_color{jd},'linewidth',2.5); hold on

end


figure(1)
for j = 1:2
    subplot(1,2,j)
    if j == 1
        legend('True','Raw Pres.','Smooth Pres.','Wavelet','location','northeast')
    end

    ylim([10^-3,3.5])
    xlim([-4,5])
    xticks([-4:2:4])
    grid on
    xlabel('$q$','interpreter','latex')
    ylabel('$p_q$','interpreter','latex')
    BBplotSettings(25,0)
end



%% Second Set of Plots

plot_color = {'g','b','r'};
figure(2);close;figure(2)
figure(3);close;figure(3)


for jd = 1:3
    inputData = inputDataVec{jd};

    % Time Series Plot
    figure(2)
    xlim([830,1000])
    subplot(3,1,jd)
    plot(Q{1,1}(:,1),Q{1,1}(:,4),'k','linewidth',2); hold on
    plot(Q{2,jd}(:,1),Q{2,jd}(:,2),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(20,1);
    if jd == 3
        xlabel('$t$','interpreter','latex');
    end
    ylabel('$q$','interpreter','latex');
    xlim([830,1000]);
%     ylim([-3.5,5]);


 % Time Series Plot
    figure(3)
    xlim([830,1000])
    subplot(3,1,jd)
    plot(Q{1,1}(:,1),Q{1,1}(:,4),'k','linewidth',2); hold on
    plot(Q{1,jd}(:,1),Q{1,jd}(:,2),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(20,1);
    if jd == 3
        xlabel('$t$','interpreter','latex');
    end
    ylabel('$q$','interpreter','latex');
    xlim([830,1000]);
end
