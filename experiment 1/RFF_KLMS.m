%RFF-KLMS,�����Ƶ���ԭ����ѡ��Ϊ����p*2/D�ȽϽӽ���������Ϊ�Ƶ���������һ����ƽ����1/D,��cosǰ���ϵ��sqrt(2),��Ϊ�ڻ�������2��
function [mse_te,time] = RFF_KLMS(X,T,X_te1,T_te1,X_te2,T_te2,w,D,mu)
N_tr = size(X,2);
N_te = size(X_te1,2);
b = rand(1,D)*2*pi;
omega = zeros(1,D);
e = zeros(N_tr,1);
mse_te = zeros(N_tr,1);
time = zeros(N_tr,1);
%start
for i = 1:N_tr
    tic;
    %trainning
     psi_x = cos(X(:,i)'*w+b);
     y_prediction = omega*psi_x';
     e(i) = T(i)-y_prediction;
     omega = omega + mu*e(i)*psi_x;
     %teating
     time(i) = toc;
     if i<N_tr/2
         psi_x = cos(X_te1'*w+ones(N_te,1)*b);
         text_y_pr = (omega*psi_x')';
         err = T_te1-text_y_pr';
         mse_te(i) = mean(err.^2);
     else
         psi_x = cos(X_te2'*w+ones(N_te,1)*b);
         text_y_pr = (omega*psi_x')';
         err = T_te2-text_y_pr';
         mse_te(i) = mean(err.^2);
     end

end
end
%=========   end of RFF-KLMS  ================