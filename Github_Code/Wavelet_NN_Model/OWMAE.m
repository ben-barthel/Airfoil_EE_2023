function [mae,owmae] = OWMAE(y,py,yhat)
% Computes Mean Abolute Error and Output-Weighted Mean Aboslute Error


py_hat = interp1(py(:,1),py(:,2),yhat,'linear','extrap');
py_hat(py_hat<10^-5) = 10^-5;
n = length(y);
mae = sum(abs(y-yhat))/n;
owmae = sum(abs(y-yhat)./py_hat)/n;

% 
% figure
% plot(py(:,1),py(:,2),'k','linewidth',5);hold on;
% plot(yhat,py_hat,'.r','markersize',10);hold on;
% BBplotSettings(25,0)




end