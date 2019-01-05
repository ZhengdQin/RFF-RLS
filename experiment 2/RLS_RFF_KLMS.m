%RFF-KLMS,根据推导的原理步长选择为步长p*2/D比较接近其结果，因为推导过程中有一个求平均的1/D,和cos前面的系数sqrt(2),因为内积所以是2；
function mse_te = RLS_RFF_KLMS(X,T,X_te,T_te,w,D,regularizationFactor,forgetFctor)
N_tr = size(X,2);
N_te = size(X_te,2);
b = rand(1,D)*2*pi;
omega = zeros(D,1);
e = zeros(N_tr,1);
mse_te = zeros(N_tr,1);
P = (1/regularizationFactor)*eye(D);
%start
for i = 1:N_tr
    %trainning
     psi_x = cos(X(:,i)'*w+b);
     y_prediction = omega'*psi_x';
     e(i) = T(i)-y_prediction;
     %omega = omega + mu*e(i)*psi_x;
     [omega,Pnew] = LSE(psi_x',omega,e(i),P,forgetFctor);
     P = Pnew;
     %teating
         psi_x = cos(X_te'*w+ones(N_te,1)*b);
         text_y_pr = (omega'*psi_x')';
         err = T_te-text_y_pr;
         mse_te(i) = mean(err.^2);
end
end
%=========   end of RFF-KLMS  ================