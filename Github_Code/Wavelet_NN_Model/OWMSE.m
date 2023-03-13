function [mae,owmae] = OWMSE(y,py,yhat)
% Computes Mean Square Error and Output-Weighted Mean Square Error


py_hat = interp1(py(:,1),py(:,2),yhat,'linear','extrap');
py_hat(py_hat<10^-5) = 10^-5;

n = length(y);
mae = sum(abs((y-yhat).^2))/n;
owmae = sum(abs((y-yhat).^2)./py_hat)/n;

% 
% figure
% plot(py(:,1),py(:,2),'k','linewidth',5);hold on;
% plot(yhat,py_hat,'.r','markersize',10);hold on;
% BBplotSettings(25,0)
% 



end