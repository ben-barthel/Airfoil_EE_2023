clear all
close all
clc
computeWT = 0;
saveWT = 0;
% Subset for Wavelet transform
iw1 = 1;
iw2 = 1000;
%% Vorticity snapshots
t = readNPY('time_snap_shot.npy');
x = readNPY('x_snap_shot.npy');
y = readNPY('y_snap_shot.npy');
w = readNPY('vorticity_snap_shot.npy');
xwing = readNPY('wing_x.npy');
ywing = readNPY('wing_y.npy');
% Drag  Coefficient
[tq,P,q] = load_airfoil_data(0);
[~,Ps,~] = load_airfoil_data(1);

ny = length(y);
nx = length(x);
nt = length(t);

load waveletDB.mat
gamma = waveletDB{35};
% Markers for subplots
nIDs = 26;
alphabet = ('a':'z').';
chars = num2cell(alphabet(1:nIDs));
chars = chars.';
charlbl = strcat('(',chars,')'); % {'(a)','(b)','(c)','(d)'}

%% Find snapshots and create mask
[X,Y] = meshgrid(x,y);
[mask_wing,mask_surf] = inpolygon(X,Y,xwing,ywing);
mask_wing = 1 - mask_wing;


[~,ind1] = min(abs(t-904));
[~,ind2] = min(abs(t-925));
[~,indm] = min(abs(t-921.58));
[~,indb1] = min(abs(t-911.22));
[~,indb2] = min(abs(t-931.4));
ind_test = [ind1,indb1,indm,ind2];

ind_test2 = linspace(indb1,indb2,9);
ind_test2 = floor(ind_test2);
ind_test2(5) = indm;


ncon = [400,401,400,401];
dt = t(2)-t(1);
dx = x(2)-x(1);
dy = y(2)-y(1);
%% Wavelet Transform
if computeWT
ww = w(:,:,iw1:iw2);
tt = t(iw1:iw2);
ntt = length(tt);
nf = 81;
Fs = 1/(dt);
ff = linspace(0,2,nf);
[~,ivs] = min(abs(ff-1.44));
ff(ivs) = 1.44;
[TT,FF] = meshgrid(tt,ff);
WT_full = zeros(nx,ny,nf); 

    name = 'amor';
    for jx = 1:nx
        if jx ==1
            disp('Starting Wavelet Transform...')
        end
        if mod(jx,100) == 0
            disp(['Progress = ',num2str(jx),'/800'])
        end
        for jy = 1:ny
            % Wavelet Transform
            data_in = squeeze(ww(jx,jy,:));
%             data_in = interp1(tt,data_in,tf,'linear','extrap');
            [wt,fwt] = cwt(data_in,name,Fs);
            WT = interp2(tt,fwt,wt,TT,FF);
            [~,ind] = min(abs(ff-0.4) );
            [~,ind2] = min(abs(ff-1.44) );
            WT04(jx,jy,:) = (WT(ind,:));
            WT144(jx,jy,:) = (WT(ind2,:));
            WT_full(jx,jy,:) = trapz(abs(WT).^2,2)*dt;
        end
    end
   vort_spg = squeeze(trapz(trapz(abs(WT_full),1)*dx,2)*dy); 
else
    load vorticity_Wavelet_Transform_f04_Nt1000.mat

end
%% Spectrogram

fff = linspace(0,2,200);

figure(926);close;figure(926)
plot(ff,abs(vort_spg),'k','linewidth',2)
BBplotSettings(25,1);
xlabel('$f$','interpreter','latex')
ylabel('$\|\hat{\Omega}\|$','interpreter','latex')
set(gca,'YScale','log')
%% Initial Plots
figure(2)
snap = squeeze(w(:,:,indm))';
% snap(abs(snap)<0.05) = 0;
contourf(x,y,snap,400,'LineStyle','none');
patch(xwing,ywing,'k')
BBplotSettings(25,0);
colormap('REDBLUE')
colorbar;
caxis([-50,50])
xlim([-0.3,2]);ylim([-0.5,0.5]);


figure(3);close;figure(3)
for j = 1:4

    snap = squeeze(w(:,:,ind_test(j)))';
    snap(abs(snap)<0.1) = 0;
    subplot(2,2,j)
    contourf(x,y,snap,ncon(j),'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    xlim([-0.3,2]);ylim([-0.5,0.5]);
    %     colorbar;
    caxis([-50,50])
    if j == 3
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test(j))))],'Interpreter','latex')
    end
end


%% Traversing of one EE Plot


figure(4);close;figure(4)
for j = 1:length(ind_test2)
    snap = squeeze(w(:,:,ind_test2(j)))';
    snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,snap,201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
    xlim([-0.3,2]);ylim([-0.5,0.5]);
end


figure(44);close;figure(44);
subplot(3,3,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([911,933])
ylim([-2,4.5])
BBplotSettings(25,0)
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
for j = 2:length(ind_test2)
    [~,indpqj] = min(abs(tq-t(ind_test2(j))));
    plot(tq(indpqj),q(indpqj),'.r','markersize',35)
end
for j = 2:length(ind_test2)
    snap = squeeze(w(:,:,ind_test2(j)))';
    snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,snap,201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
%     xlim([-0.3,2]);ylim([-0.5,0.5]);
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end
%% Drag Coefficent Plot

% lets analyze extreme event from t = 904 - t = 925
figure(1);close;figure(1);
plot(tq,q,'k','LineWidth',2);hold on
xlim([911,931])
BBplotSettings(25,0)
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
for j = 1:length(ind_test2)
    [~,indpqj] = min(abs(tq-t(ind_test2(j))));
    figure(1)
    plot(tq(indpqj),q(indpqj),'.r','markersize',35)
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
figure(6);close;figure(6)
for j = 1:length(ind_test2)
    [~,indW] = min(abs(tt - t(ind_test2(j))));
    snap = squeeze(WT04(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
    xlim([-0.3,2]);ylim([-0.5,0.5]);
    %     xlim([0.4,0.9]);ylim([-0.1,0.1]);
end

%% Extreme Event Mode - BL
figure(7);close;figure(7)
for j = 1:length(ind_test2)
    [~,indW] = min(abs(tt - t(ind_test2(j))));
    snap = squeeze(WT04(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end


%% Vortex Shedding Mode - Farfield
figure(66);close;figure(66)
for j = 1:length(ind_test2)
    [~,indW] = min(abs(tt - t(ind_test2(j))));
    snap = squeeze(WT144(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
    colorbar
    xlim([-0.3,2]);ylim([-0.5,0.5]);
    %     xlim([0.4,0.9]);ylim([-0.1,0.1]);
end

%% Vortex shedding Mode - BL
figure(77);close;figure(77)
for j = 1:length(ind_test2)
    [~,indW] = min(abs(tt - t(ind_test2(j))));
    snap = squeeze(WT144(:,:,indW))';
    %     snap(abs(snap)<0.1) = 0;
    subplot(3,3,j)
    contourf(x,y,mask_wing.*real(snap),201,'LineStyle','none');
    patch(xwing,ywing,'k')
    BBplotSettings(25,0);
    colormap('REDBLUE')
    if j ==5
        title(['Extreme Event: ','$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    else
        title(['$t = $',' ',num2str(floor(t(ind_test2(j))))],'Interpreter','latex')
    end
    caxis([-50,50])
%     colorbar
    xlim([0.4,0.9]);ylim([-0.1,0.1]);
end

%% Wavelet Scalar Metrics
% [~,ind0] = min(abs(tt - t(ind_test2(1))));
[~,ind0] = min(abs(tt - 911));
for j = 1:length(tt)

    % EE Mode
    snap0 = (squeeze(WT04(:,:,ind0))');
    snap = (squeeze(WT04(:,:,j))');
    % norm
    W2dx = trapz(conj(snap).*mask_wing.*snap,1)*dx;
    W2dxdy = trapz(W2dx,2)*dy;
    wavelet_norm(j) = abs(W2dxdy);
    % projection
    Wp2dx = trapz(conj(snap).*mask_wing.*snap0,1)*dx;
    Wp2dxdy = trapz(Wp2dx,2)*dy;
    wavelet_proj(j) = abs(Wp2dxdy);
    
    clear snap snap0

    % VS Mode
    snap0 = (squeeze(WT144(:,:,ind0))');
    snap = abs(squeeze(WT144(:,:,j))');
    % norm
    W2dx = trapz(conj(snap).*mask_wing.*snap,1)*dx;
    W2dxdy = trapz(W2dx,2)*dy;
    vs_mode_norm(j) = abs(W2dxdy);
    % projection
    Wp2dx = trapz(conj(snap).*mask_wing.*snap0,1)*dx;
    Wp2dxdy = trapz(Wp2dx,2)*dy;
    vs_wavelet_proj(j) = abs(Wp2dxdy);
    
end
%% Wavelet Scalar Metric Plots
mi = 4;
figure(8);close;figure(8);
subplot(2,1,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(tt),max(tt)])
xlim([810,1000])
% ylim([-3,5])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right

plot(tt,wavelet_proj./max(wavelet_proj(ind0)),'.-b','linewidth',1.5,'MarkerSize',26,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(tt,vs_wavelet_proj./max(vs_wavelet_proj(ind0)),'<-r','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi:length(tt)]); hold on;

set(gca,'YColor','k')
ylabel('$r$','Interpreter','latex')
% legend('$q(t)$','$|\langle \hat{\Omega}(t_0),\hat{\Omega}(t) \rangle|$', '$\|\hat{\Omega}(t)\|^2$', '$|\gamma(s,t)|_{s=0.35}$','Interpreter','latex')
BBplotSettings(25,1);
subplot(2,1,2)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(tt),max(tt)])
xlim([810,1000])
% ylim([-3,5])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(tt,wavelet_norm./max(wavelet_norm(ind0)),'.-b','linewidth',1.5,'MarkerSize',26,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(tt,vs_mode_norm./max(vs_mode_norm(ind0)),'<-r','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(gamma.t,gamma.gm./max(gamma.gm),'s-g','linewidth',1.5,'MarkerSize',7,'MarkerIndices',[1:mi*20:length(gamma.t)]); hold on;
% ylim([0,1.2])
set(gca,'YColor','k')
ylabel('$\|\hat{\Omega}_e\|^2 ~\ ,\|\hat{\Omega}_v\|^2 ~\ ,~\ |\gamma|$','Interpreter','latex')
BBplotSettings(25,1);

% 3 subplot Version

figure(99);close;figure(99)
subplot(3,1,2)
plot(tt,wavelet_norm,'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi*2:length(tt)]); hold on;
plot(tt,vs_mode_norm,'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(tt)]); hold on;
xlim([810,1000])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$\|\hat{\Omega}\|^2 $','Interpreter','latex')

subplot(3,1,1)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(tt),max(tt)])
xlim([810,1000])
% ylim([-3,5])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(tt,wavelet_proj./max(wavelet_proj(ind0)),'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(tt,vs_wavelet_proj./max(vs_wavelet_proj(ind0)),'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(tt)]); hold on;
set(gca,'YColor','k')
ylabel('$r$','Interpreter','latex')
% legend('$q(t)$','$|\langle \hat{\Omega}(t_0),\hat{\Omega}(t) \rangle|$', '$\|\hat{\Omega}(t)\|^2$', '$|\gamma(s,t)|_{s=0.35}$','Interpreter','latex')
BBplotSettings(25,1);

subplot(3,1,3)
plot(tq,q,'k','LineWidth',2);hold on
xlim([min(tt),max(tt)])
xlim([810,1000])
% ylim([-3,5])
BBplotSettings(25,1);
xlabel('$t$','Interpreter','latex')
ylabel('$q$','Interpreter','latex')
yyaxis right
plot(tt,wavelet_norm./max(wavelet_norm(ind0)),'.-b','linewidth',1.5,'MarkerSize',25,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(tt,vs_mode_norm./max(vs_mode_norm(ind0)),'<-r','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi:length(tt)]); hold on;
plot(gamma.t,gamma.gm./max(gamma.gm),'s-g','linewidth',1.5,'MarkerSize',5,'MarkerIndices',[1:mi*20:length(gamma.t)]); hold on;
% ylim([0,1.2])
set(gca,'YColor','k')
ylabel('$\|\hat{\Omega}\|^2  ,~\ |\gamma|$','Interpreter','latex')
BBplotSettings(25,1);

% Mask
% figure(1234)
% subplot(2,2,1)
% contourf(x,y,-mask_wing,'LineStyle','none')
% colormap('redblue')
% caxis([-1,1])
% subplot(2,2,2)
% contourf(x,y,-mask_surf,'LineStyle','none')
% colormap('redblue')
% caxis([-1,1])
%% Save
if saveWT
    filename = ['vorticity_Wavelet_Transform_f04_Nt',num2str(length(tt)),'.mat'];
    save(filename,'t','tt','ff','x','y','WT04','WT144','vort_spg','-v7.3');disp('wavelet data saved')
end
