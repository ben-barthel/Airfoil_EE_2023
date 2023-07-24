clear all
close all
clc
%% Inputs
tau = 7;
Nepoch = 200;
Ntest = 10;
lossTypeVec = {'MAE','OWMAE'};
%% Load Data
[Q,PQ] = wavelet_NN_load_active_search_results(tau,Nepoch,Ntest,lossTypeVec);
%% Plots
plotcolors = {'black','green','blue','red','magenta','red','cyan','magenta'};
for jl = 1:2
    % PDF Plots
    figure(1);
    subplot(1,2,jl)
    semilogy(PQ{1}(:,1),PQ{1}(:,3),'k','linewidth',2); hold on
    colororder(plotcolors)
    for j = 1:3
        lsty = '-';
        [pqj,qjv] = ksdensity(squeeze(Q{j}(:,2)));

        subplot(1,2,jl)
        semilogy(PQ{j}(:,1),PQ{j}(:,2),'linewidth',2,'linestyle',lsty); hold on
        if jl == 1
            legend('True','Ref.','IU','PW','location','northwest','Orientation','horizontal')
        end
        subplot(1,2,jl)
        ylim([2*10^-2,1.2])
        xlim([-4,5])
        xticks([-4:2:4])
        grid on
        xlabel('$q$','interpreter','latex')
        ylabel('$p_q$','interpreter','latex')
        set(gca,'fontsize',25,'fontname','times')
    end


    % Time Series Plots
    figure(2);
    subplot(2,1,jl)
    plot(Q{1}(:,1),Q{1}(:,4),'k','linewidth',2); hold on
    colororder(plotcolors)
    lsty = '-';
    for j = 1:3
        plot(Q{j}(:,1),Q{j}(:,2),'linewidth',2,'linestyle',lsty); hold on
    end
    if jl ==1
        legend('True','Ref.','IU','PW','location','southwest','Orientation','horizontal')
    end
    set(gca,'fontsize',25,'fontname','times')
    xlim([830,1000])
    ylim(([-3,4.5]))
    ylabel('$q$','interpreter','latex')
    xlabel('$t$','interpreter','latex')

    figure(jl+2)
    for j = 1:3
        subplot(3,1,j)
        colororder(plotcolors)
        plot(Q{1}(:,1),Q{1}(:,4),'k','linewidth',2); hold on
        plot(Q{j}(:,1),Q{j}(:,2)-Q{j}(:,3),plotcolors{j+1},'linewidth',1); hold on
        plot(Q{j}(:,1),Q{j}(:,2)+Q{j}(:,3),plotcolors{j+1},'linewidth',1); hold on
        x = Q{j}(:,1);y1 = Q{j}(:,2)-Q{j}(:,3); y2 = Q{j}(:,2)+Q{j}(:,3);
        patch([x' fliplr(x')], [y1' fliplr(y2')], plotcolors{j+1},'facealpha',0.25)
        xlim([830,1000])
        ylim(([-3,4.5]))
        set(gca,'fontsize',20,'fontname','times')
        ylabel('$q$','interpreter','latex')
        xlabel('$t$','interpreter','latex')
    end

end

