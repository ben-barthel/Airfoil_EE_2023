function [] = OMI_plot(OPT_IND,s)
plot_color ={'-sb','->r','-xg'};
for m = 1:2
    figure(1);
    subplot(1,2,m)
    for k = 1:Nsens
        h = k*ones(k,1)';
        d = s(OPT_IND{m}(1:k));
        plot(d,h,plot_color{m},'marker','.','markersize',55,'linestyle','none');hold on;
        xticks([0,0.25,0.5,0.75,1.0]);grid on;
        BBplotSettings(30,1)
        xlabel('$s$','interpreter','latex');
        ylabel('$N$','interpreter','latex');
        xlim([0,1])
        yticks([1,2,3,4,5,6])
    end
end
end