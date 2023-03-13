clear all
close all
clc
%% Inputs
lossTypeVec = {'MAE','OWMAE'};
inputDataVec = {'rawP','smoothP','d2wavelet'};
tauv = [0,3,7,10];
%% Load Data
for jt =1:length(tauv)
    tau = tauv(jt);
    for jl = 1:2
        lossType = lossTypeVec{jl};
        for jd = 1:3
            inputData = inputDataVec{jd};

            % Test set Data: t, q_mean, q_std, q_true
            % Pdf data: val, pq_model, pq_true

            % Load
            name1 = ['../../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep201_Ntest10_',lossType,'_S_(5, 35, 65, 95)_out.npy'];
            name2 = ['../../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep201_Ntest10_',lossType,'_S_(5, 35, 65, 95)_pdf.npy'];
            Q{jl,jd,jt}  = readNPY(name1);
            PQ{jl,jd,jt} = readNPY(name2);
            

        end
    end
end
% Alpha star
alpha = readNPY('../../Basis_Test_Ensemble_d2wavelet_NN_tau10_Nep201_Ntest10(5, 35, 65, 95)_alpha_star.npy');
a_P = squeeze(alpha(1,:,:));
a_Ps = squeeze(alpha(2,:,:));
a_W = squeeze(alpha(3,:,:));

%% Metrics




for jl = 1:2
    lossType = lossTypeVec{jl};
    for jd = 1:3
        inputData = inputDataVec{jd};
        for jt = 1:length(tauv)
            % True Output
            pq(:,1) = PQ{jl,jd,jt}(:,1);
            pq(:,2) = PQ{jl,jd,jt}(:,3);
            q_true = Q{jl,jd,jt}(:,4);
            t = Q{jl,jd,jt}(:,1);
            % Model Prediction
            q = Q{jl,jd,jt}(:,2);
            % Standard & Output-Weighted Mean Absolute Error
            [e,e_ow] = OWMAE(q_true,pq,q);
            mae(jl,jd,jt) = e;
            owmae(jl,jd,jt) = e_ow;
            % Standard & Output-Weighted Mean Square Error
            [e,e_ow] = OWMSE(q_true,pq,q);
            mse(jl,jd,jt) = e;
            owmse(jl,jd,jt) = e_ow;

            if jl ==2
                % find peaks
                [pks_t{jd,jt},locs_t{jd,jt}] = findpeaks(q_true,t,'MinPeakHeight',2);
                [pks_m{jd,jt},locs_m{jd,jt}] = findpeaks(q,t,'MinPeakHeight',2);
                numPks(jd,jt) = length(locs_m{jd,jt});
                numPks_t(jd,jt) = length(locs_t{jd,jt});

                % test

                figure(1000+jd)
                subplot(2,2,jt)
                findpeaks(q_true,t,'MinPeakHeight',2,'MinPeakDistance',2.5); hold on;
                findpeaks(q,t,'MinPeakHeight',2,'MinPeakDistance',2.5);
                BBplotSettings(25,1);


            end
        end
    end
end

%% Plots
figure(1); close; figure(1)

plot_color={'o-g','o-b','o-r'};
for jd = 1:3
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
    plot(tauv,abs(numPks(jd,:)-numPks_t(jd,:)),plot_color{jd},'linewidth',2.5); hold on;
    BBplotSettings(25,1);
    ylim([0.3,1000])
    yticks([1,10,100])
    set(gca,"YScale","log")
    xlabel('$\tau$','Interpreter','latex')
    ylabel('$|N_{true}-N|$','Interpreter','latex')
end



subplot(2,2,4)
plot(tauv,a_P(1,:),'o-g','LineWidth',2.5); hold on;
plot(tauv,a_Ps(1,:),'o-b','LineWidth',2.5); hold on;
plot(tauv,a_W(1,:),'o-r','LineWidth',2.5); hold on;
BBplotSettings(25,1);
set(gca,"YScale","log")
xlabel('$\tau$','Interpreter','latex')
ylabel('$\alpha^*$','Interpreter','latex')





%% Rolling PEak metric
true_peaks = [];
plot_color={'--g','--b','--r'};
fc=[1.44,0.4,0.5;]
for jd = 1:3
    inputData = inputDataVec{jd};
    for jt = 1:length(tauv)
        % True Output
        pq(:,1) = PQ{jl,jd,jt}(:,1);
        pq(:,2) = PQ{jl,jd,jt}(:,3);
        q_true = Q{jl,jd,jt}(:,4);
        t = Q{jl,jd,jt}(:,1);
        % Model Prediction
        q = Q{jl,jd,jt}(:,2);
        % rolling over time
        for j = 3:length(t)

            tj = t(1:j);
            qj = q(1:j);
            qtj = q_true(1:j);
            if max(tj)-min(tj) < 3
                mdt = 0;
            else
                mdt = 1/fc(jd);
            end

            [pt,lt] = findpeaks(qtj,tj,'MinPeakHeight',2,'MinPeakDistance',mdt);
            [pm,lm] = findpeaks(qj,tj,'MinPeakHeight',2,'MinPeakDistance',mdt);

            model_peak_count{jd,jt}(j) = length(lm);
            true_peak_count{jd,jt}(j) = length(lt);



        end


    end
end

%%
figure(10);close;
plot_color={'<-g','>-b','o-r'};
npp = 400;
for jt = 1:length(tauv)

    for jd =2:3


        figure(10)
        subplot(2,2,jt)
        plot(Q{jl,jd,jt}(1:npp:end,1),true_peak_count{jd,jt}(1:npp:end),'k','linewidth',3); hold on;
        plot(Q{jl,jd,jt}(1:npp:end,1),model_peak_count{jd,jt}(1:npp:end),plot_color{jd},'LineWidth',1.5); hold on;
        BBplotSettings(25,1);
        set(gca,"YScale",'linear')
        xlim([830,1000]);
        %         ylim([0.1,500])
        xlabel('$t$','Interpreter','latex')
        ylabel('$N_{EE}(t)$','Interpreter','latex')
        %         title(['$\tau = ',num2str(tauv(jt)),'$'],'Interpreter','latex')

    end
end

for jt = 1:length(tauv)
    subplot(2,2,jt)
    annotation('textbox', [0.5, 0.2, 0.1, 0.1], 'String',['$\tau = ',num2str(tauv(jt)),'$'],'Interpreter','latex','FontSize',15,'EdgeColor','none')

end
