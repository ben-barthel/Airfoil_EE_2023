function [Q,PQ] = wavelet_NN_load_results(tau,Nepoch,Ntest,lossTypeVec,inputDataVec)
for jl = 1:length(lossTypeVec)
    lossType = lossTypeVec{jl};
    for jd = 1:length(inputDataVec)
        inputData = inputDataVec{jd};
        % Test set Data: t, q_mean, q_std, q_true
        % Pdf data: val, pq_model, pq_true
        % Load
        name1 = ['../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep',num2str(Nepoch),'_Ntest',num2str(Ntest),'_',lossType,'_S_(5, 35, 65, 95)_out.npy'];
        name2 = ['../Basis_Test_Ensemble_',inputData ,'_NN_tau',num2str(tau),'_Nep',num2str(Nepoch),'_Ntest',num2str(Ntest),'_',lossType,'_S_(5, 35, 65, 95)_pdf.npy'];
        Q{jl,jd}  = readNPY(name1);
        PQ{jl,jd} = readNPY(name2);

    end
end
end