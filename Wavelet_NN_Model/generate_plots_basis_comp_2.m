clear all
close all
clc
%% Inputs
lossTypeVec = {'MAE','OWMAE'};
inputDataVec = {'rawP','smoothP','wavelet'};
tauv = [0,3,7,10];
Nepoch = 250;
Ntest = 10;
%% Load Data
[a,Q,PQ] = wavelet_NN_load_tau_results(tauv,Nepoch,Ntest,lossTypeVec,inputDataVec);
%% Global Metrics
[mae,owmae,mse,owmse,numPks,numPks_t,model_peak_count,true_peak_count] = wavelet_NN_metrics(tauv,Q,PQ,lossTypeVec,inputDataVec);
%% Global Metric Plots
figure(1); close; figure(1)
plot_color = {'<-g','>-b','o-r'};
for jd = 1:length(inputDataVec)
    inputData = inputDataVec{jd};
    figure(1)
    subplot(2,2,1)
    plot(tauv,squeeze(mae(2,jd,:)),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(25,1);xlabel('$\tau$','Interpreter','latex');
    ylabel('$MAE$','Interpreter','latex');

    subplot(2,2,2)
    plot(tauv,squeeze(mse(2,jd,:)),plot_color{jd},'linewidth',2); hold on
    BBplotSettings(25,1);xlabel('$\tau$','Interpreter','latex');
    ylabel('$MSE$','Interpreter','latex');

    subplot(2,2,3)
    if jd >1
    plot(tauv,abs(numPks(jd,:)-numPks_t(jd,:)),plot_color{jd},'linewidth',2.5); hold on;
    end
    BBplotSettings(25,1);
%     ylim([0.3,1000])
%     yticks([1,10,100])
%     set(gca,"YScale","log")
    xlabel('$\tau$','Interpreter','latex')
    ylabel('$|N_{true}-N|$','Interpreter','latex')

    subplot(2,2,4)
    plot(tauv,squeeze(a(jd,1,:)),plot_color{jd},'linewidth',2.5); hold on;
    BBplotSettings(25,1);
    set(gca,"YScale","log")
    xlabel('$\tau$','Interpreter','latex')
    ylabel('$\alpha^*$','Interpreter','latex')

end


%% Time Dependent Metric Plots
figure(10); close; figure(10)
npp = 200;
for jt = 1:length(tauv)
    for jd = 1:length(inputDataVec)
        inputData = inputDataVec{jd};
        % don't plot raw pressure results
        if strcmp(inputData,'rawP')
            continue
        end
        figure(10)
        subplot(2,2,jt)
        plot(Q{2,jd,jt}(:,1),true_peak_count{jd,jt},'k','linewidth',3,'MarkerIndices',1:npp:length(Q{2,jd,jt}(:,1))); hold on;
        plot(Q{2,jd,jt}(:,1),model_peak_count{jd,jt},plot_color{jd},'LineWidth',1.5,'MarkerIndices',1:npp:length(Q{2,jd,jt}(:,1))); hold on;
        BBplotSettings(25,1);
        set(gca,"YScale",'linear')
        xlim([830,max(Q{2,jd,jt}(:,1))]);
        xlabel('$t$','Interpreter','latex')
        ylabel('$N_{EE}$','Interpreter','latex')
    end
    annotation('textbox', [0.5, 0.2, 0.1, 0.1], 'String',['$\tau = ',num2str(tauv(jt)),'$'],'Interpreter','latex','FontSize',15,'EdgeColor','none')
end
