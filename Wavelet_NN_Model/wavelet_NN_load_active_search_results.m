function [Q,PQ] = wavelet_NN_load_active_search_results(tau,Nepoch,Ntest,lossTypeVec)
for jl = 1:2
 
    % Test set Data: t, q_mean, q_std, q_true
    % Pdf data: val, pq_model, pq_true
    NS = 7;
    lossType = lossTypeVec{jl};
    tag = {'ref','IU','PW'}; 
    plotcolors = {'black','green','blue','red','magenta','red','cyan','magenta'};
    name0 = ['../Ensemble_wavelet_NN_tau',num2str(tau),'_Nep',num2str(Nepoch),'_Ntest',num2str(Ntest),'_',lossType,'_'];

    for j = 1:3
        name1 = [name0,tag{j},'_out.npy'];
        name2 = [name0,tag{j},'_pdf.npy'];
        Q{j} = readNPY(name1);
        PQ{j} = readNPY(name2);

    end

end


end