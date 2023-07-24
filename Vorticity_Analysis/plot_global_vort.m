clear all
close all
clc
%% Load Data
[x,y,t,w,xwing,ywing] = load_vorticity();
[mask_wing] = airfoil_mask(x,y,xwing,ywing);
load ../vorticity_Wavelet_Transform_f04_Nt1000.mat
[tq,~,q] = load_airfoil_data(0);
load('../waveletDB.mat')
gamma = waveletDB{35};
%% Wavelet Scalar Metrics
[ee_mode_norm,ee_mode_proj,vs_mode_norm,vs_mode_proj] = global_metrics(WT04,WT144,x,y,t,mask_wing);
%% Snapshots for plot
[~,indm] = min(abs(t-921.58));
[~,indb1] = min(abs(t-911.22));
[~,indb2] = min(abs(t-931.4));
ind_test = linspace(indb1,indb2,9);
ind_test = floor(ind_test);ind_test2(5) = indm;


ind_test = ind_test([2,5,8]);
%% Full Solution - Farfield
% figure(4);close;figure(4)
% for j = 1:length(ind_test)
%     snap = squeeze(w(:,:,ind_test(j)))';
%     snap(abs(snap)<0.1) = 0;
%     subplot(3,3,j)
%     contourf(x,y,snap,201,'LineStyle','none');
%     patch(xwing,ywing,'k')
%     BBplotSettings(25,0);
%     colormap('REDBLUE')
%     if j ==5
%         title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     else
%         title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     end
%     caxis([-50,50])
%     xlim([-0.3,2]);ylim([-0.5,0.5]);
% end
%% Snapshots
figure(11);close;figure(11);
subplot(3,3,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([911,933])
ylim([-2,4.5])
BBplotSettings(25,0)
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
for j = 1:length(ind_test)
    [~,indpqj] = min(abs(tq-t(ind_test(j))));
    plot(tq(indpqj),q(indpqj),'.r','markersize',35)
end

%% Full Solution - BL
figure(1);close;figure(1);
% subplot(3,3,1)
% plot(tq,q,'k','LineWidth',2);hold on
% xlim([911,933])
% ylim([-2,4.5])
% BBplotSettings(25,0)
% xlabel('$t$','Interpreter','latex')
% ylabel('$q$','Interpreter','latex')
% for j = 1:length(ind_test)
%     [~,indpqj] = min(abs(tq-t(ind_test(j))));
%     plot(tq(indpqj),q(indpqj),'.r','markersize',35)
% end
for j = 1:length(ind_test)
    snap = squeeze(w(:,:,ind_test(j)))';
    snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,snap,201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j == 5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
%     xlim([-0.3,2]);ylim([-0.5,0.5]);
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end
%% Multiple EE peaks

tpv = [812.02,827.33,849.64,864.62,883.97,901.46,921.58,938.99,956.37,974.32,995.8];

figure(5);close;figure(5)
for j = 1:9
    [~,indpj] = min(abs(t-tpv(j)));
    snap = squeeze(w(:,:,indpj))';
    snap(abs(snap)<0.11) = 0;
    figure(5)
    subplot(3,3,j)
    contourf(x,y,snap,200,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    xlim([-0.3,2]);ylim([-0.5,0.5]);
    title(['Extreme Event Peak: ','$t = $',' ',num2str(floor(tpv(j)))],'Interpreter','latex')
    caxis([-50,50])


end




%% Extreme Event Mode - Farfield
% figure(6);close;figure(6)
% for j = 1:length(ind_test)
%     [~,indW] = min(abs(t - t(ind_test(j))));
%     snap = squeeze(WT04(:,:,indW))';
%     %     snap(abs(snap)<0.1) = 0;
%     subplot(3,3,j)
%     contourf(x,y,real(snap),201,'LineStyle','none');
%     patch(xwing,ywing,'k')
%     BBplotSettings(25,0);
%     colormap('REDBLUE')
%     if j ==5
%         title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     else
%         title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     end
%     caxis([-50,50])
%     xlim([-0.3,2]);ylim([-0.5,0.5]);
%     %     xlim([0.4,0.9]);ylim([-0.1,0.1]);
% end

%% Extreme Event Mode - BL
figure(1)
for j = 1:length(ind_test)
    [~,indW] = min(abs(tt - t(ind_test(j))));
    snap = squeeze(WT04(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j+3)
    contourf(x,y,real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
%     if j ==5
%         title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     else
%         title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     end
    caxis([-50,50])
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end


%% Vortex Shedding Mode - Farfield
% figure(66);close;figure(66)
% for j = 1:length(ind_test)
%     [~,indW] = min(abs(t - t(ind_test(j))));
%     snap = squeeze(WT144(:,:,indW))';
%     %     snap(abs(snap)<0.1) = 0;
%     subplot(3,3,j)
%     contourf(x,y,real(snap),201,'LineStyle','none');
%     patch(xwing,ywing,'k')
%     BBplotSettings(25,0);
%     colormap('REDBLUE')
%     if j ==5
%         title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     else
%         title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     end
%     caxis([-50,50])
%     colorbar
%     xlim([-0.3,2]);ylim([-0.5,0.5]);
%     %     xlim([0.4,0.9]);ylim([-0.1,0.1]);
% end

%% Vortex shedding Mode - BL
figure(1)
for j = 1:length(ind_test)
    [~,indW] = min(abs(tt - t(ind_test(j))));
    snap = squeeze(WT144(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j+6)
    contourf(x,y,mask_wing.*real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
%     if j == 5
%         title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     else
%         title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
%     end
    caxis([-50,50])
%     colorbar
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end
%% Wavelet Scalar Metric Plots
mi = 4;
figure(8);close;figure(8);
subplot(2,1,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(t),max(t)])
xlim([810,1000])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(t,ee_mode_proj./max(ee_mode_proj(ind0)),'.-b','linewidth',1.5,'MarkerSize',26,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(t,vs_mode_proj./max(vs_mode_proj(ind0)),'<-r','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi:length(t)]); hold on;

set(gca,'YColor','k')
ylabel('$r$','Interpreter','latex')
BBplotSettings(25,1);
subplot(2,1,2)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(t),max(t)])
xlim([810,1000])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(t,ee_mode_norm./max(ee_mode_norm(ind0)),'.-b','linewidth',1.5,'MarkerSize',26,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(t,vs_mode_norm./max(vs_mode_norm(ind0)),'<-r','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(gamma.t,gamma.gm./max(gamma.gm),'s-g','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi*20:length(gamma.t)]); hold on;
set(gca,'YColor','k')
ylabel('$\|\hat{\Omega}_e\|^2 ~\ ,\|\hat{\Omega}_v\|^2 ~\ ,~\ |\gamma|$','Interpreter','latex')
BBplotSettings(25,1);

% 3 subplot Version
figure(99);close;figure(99)
subplot(3,1,2)
plot(t,ee_mode_norm,'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi*2:length(t)]); hold on;
plot(t,vs_mode_norm,'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(t)]); hold on;
xlim([810,1000])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$\|\hat{\Omega}\|^2 $','Interpreter','latex')

subplot(3,1,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(t),max(t)])
xlim([810,1000])

BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(t,ee_mode_proj./max(ee_mode_proj(ind0)),'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(t,vs_mode_proj./max(vs_mode_proj(ind0)),'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(t)]); hold on;
set(gca,'YColor','k')
ylabel('$r$','Interpreter','latex')
BBplotSettings(25,1);

subplot(3,1,3)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(t),max(t)])
xlim([810,1000])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(t,ee_mode_norm./max(ee_mode_norm(ind0)),'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(t,vs_mode_norm./max(vs_mode_norm(ind0)),'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(t)]); hold on;
plot(gamma.t,gamma.gm./max(gamma.gm),'s-g','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi*20:length(gamma.t)]); hold on;
set(gca,'YColor','k')
ylabel('$\|\hat{\Omega}\|^2  ,~\ |\gamma|$','Interpreter','latex')
BBplotSettings(25,1);

