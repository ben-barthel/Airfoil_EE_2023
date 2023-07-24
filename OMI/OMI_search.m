function [OPT_IND] = OMI_search(I_pq,I_pq_EE,I_PP,s)

for m = 1:2
    % Acquisition Function
    switch m
        case 1
            I_al = smooth(I_Pq,window);
        case 2
            I_al = smooth(I_Pq_EE,window);
    end


    % Global Optimal
    jopt = zeros(100,1);
    [~,ind] = max(I_al);
    jopt(1) = ind;
    locs(1) = s(ind);inds(1) = ind;
    den = I_PP(:,ind);

    % Next Optimal
    for k = 1:Nsens-1

        % Find maximum
        den2 = (den/(k+1));
        a = smooth(I_al./den2,window);
        [~,ind] = max(a);

        % Update
        jopt(ind) = 1;
        den = den +  denA(:,ind);
        locs(k+1) = s(ind);
        inds(k+1) = (ind);
    end
    OPT_IND{m} = inds;
end
disp('Active Learning Complete')
end