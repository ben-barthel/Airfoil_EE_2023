function [mae,owmae,mse,owmse,numPks,numPks_t,model_peak_count,true_peak_count] = wavelet_NN_metrics(tauv,Q,PQ,lossTypeVec,inputDataVec)
%% Global Metrics
warning('off','signal:findpeaks:largeMinPeakHeight');
fc = [1.44,0.4,0.4];

for jl = 1:length(lossTypeVec)
    for jd = 1:length(inputDataVec)
        for jt = 1:length(tauv)
            % True Output
            pq(:,1) = PQ{jl,jd,jt}(:,1);
            pq(:,2) = PQ{jl,jd,jt}(:,3);
            q_true = Q{jl,jd,jt}(3:end,4);
            t = Q{jl,jd,jt}(3:end,1);
            [~,ind_end] = min(abs(t-1000)); % only use up to t = 1000
            % Model Prediction
            q = Q{jl,jd,jt}(3:end,2);
            % Standard & Output-Weighted Mean Absolute Error
            [e,e_ow] = OWMAE(q_true,pq,q);
            mae(jl,jd,jt) = e;
            owmae(jl,jd,jt) = e_ow;
            % Standard & Output-Weighted Mean Square Error
            [e,e_ow] = OWMSE(q_true,pq,q);
            mse(jl,jd,jt) = e;
            owmse(jl,jd,jt) = e_ow;
            % Extreme Event Number Metric
            if strcmp(lossTypeVec{jl},'OWMAE')
                % find peaks
                mdt = 1/fc(jd);
                [pks_t{jd,jt},locs_t{jd,jt}] = findpeaks(q_true(1:ind_end),t(1:ind_end),'MinPeakHeight',2,'MinPeakDistance',mdt);
                [pks_m{jd,jt},locs_m{jd,jt}] = findpeaks(q(1:ind_end),t(1:ind_end),'MinPeakHeight',2,'MinPeakDistance',mdt);
                numPks(jd,jt) = length(locs_m{jd,jt});
                numPks_t(jd,jt) = length(locs_t{jd,jt});
            end
        end
    end
end

%% Rolling Peak metric
for jd = 1:length(inputDataVec)
    for jt = 1:length(tauv)
        % True Output
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
            % Find Peaks
            [pt,lt] = findpeaks(qtj,tj,'MinPeakHeight',2,'MinPeakDistance',mdt);
            [pm,lm] = findpeaks(qj,tj,'MinPeakHeight',2,'MinPeakDistance',mdt);

            model_peak_count{jd,jt}(j) = length(lm);
            true_peak_count{jd,jt}(j) = length(lt);
        end
      
    end
end


end