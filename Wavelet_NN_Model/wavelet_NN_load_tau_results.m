function [a,Q,PQ] = wavelet_NN_load_tau_results(tauv,Nepoch,Ntest,lossTypeVec,inputDataVec)
%% Load Data
for jt = 1:length(tauv)
    tau = tauv(jt);
    for jl = 1:length(lossTypeVec)
        lossType = lossTypeVec{jl};
        for jd = 1:length(inputDataVec)
            inputData = inputDataVec{jd};
            % Test set Data: t, q_mean, q_std, q_true
            % Pdf data: val, pq_model, pq_true
            name1 = ['../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep',num2str(Nepoch),'_Ntest',num2str(Ntest),'_',lossType,'_S_(5, 35, 65, 95)_out.npy'];
            name2 = ['../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep',num2str(Nepoch),'_Ntest',num2str(Ntest),'_',lossType,'_S_(5, 35, 65, 95)_pdf.npy'];
            Q{jl,jd,jt}  = readNPY(name1);
            PQ{jl,jd,jt} = readNPY(name2);
        end
    end
end
% Alpha star
alpha = readNPY('../Basis_Test_Ensemble_NN_tau10_Nep201_Ntest10(5, 35, 65, 95)_alpha_star.npy');
a(1,:,:) = squeeze(alpha(1,:,:));
a(2,:,:) = squeeze(alpha(2,:,:));
a(3,:,:) = squeeze(alpha(3,:,:));

end