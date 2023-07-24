function [WT04,WT144,ff] = global_WT(w,x,y,t)
ny = length(y);
nx = length(x);
dt = t(2)-t(1);
nf = 81;
Fs = 1/(dt);
ff = linspace(0,2,nf);
[~,ivs] = min(abs(ff-1.44));
ff(ivs) = 1.44;
[TT,FF] = meshgrid(t,ff);
WT_full = zeros(nx,ny,nf);
disp('Starting Wavelet Transform...')
name = 'amor';
for jx = 1:nx
    if jx ==1
%         disp('Starting Wavelet Transform...')
    end
    for jy = 1:ny
        % Wavelet Transform
        data_in = squeeze(w(jx,jy,:));
        [wt,fwt] = cwt(data_in,name,Fs);
        WT = interp2(t,fwt,wt,TT,FF);
        [~,ind] = min(abs(ff-0.4) );
        [~,ind2] = min(abs(ff-1.44) );
        WT04(jx,jy,:) = (WT(ind,:));
        WT144(jx,jy,:) = (WT(ind2,:));

    end
end

end