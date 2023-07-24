function [WT,WT04,tt,ff] = surface_pressure_WT(t,Pin,sensor_tag)
% Wavelet Transform
j = sensor_tag;
Fs = 1/(t(2)-t(1));
tt = linspace(min(t),max(t),1000);
ff = linspace(0,2,81);
[TT,FF] = meshgrid(tt,ff);
data_in_s = Pin(j,:); data_in_s  = data_in_s - mean(data_in_s);
[wt,fwt] = cwt(data_in_s,Fs);
WT = interp2(t,fwt,wt,TT,FF);
[~,ind] = min(abs(ff-0.4) );
WT04 = abs(WT(ind,:));
disp('Wavelet Transform Computation Complete')

end