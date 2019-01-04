
function [mse_te_k_Q,Center,Center_Num,e_k_Q,e_changeAll,e_varyAll,quantzNum,time] = QKLMS(Input_train,Desire_train,Input_test1,Desire_test1,Input_test2,Desire_test2,lr_k_Q,radius_1,initialdelta)

Dimension = size(Input_train,1); 
N_tr = size(Input_train,2);
N_te = size(Input_test1,2);

%init QKLMS
e_k_Q = zeros(N_tr,1);
e_tr_Q = zeros(N_tr,1);
y_Q = zeros(N_tr,1);
mse_te_k_Q = zeros(N_tr,1);
Center = zeros(Dimension,N_tr);
kernelsize = 2*initialdelta^2;
e_quantz = zeros(N_tr,N_tr,2);
e_vary = zeros(N_tr,N_tr);
quantzNum = zeros(N_tr,1);

% n=1 init
e_k_Q(1) = Desire_train(1);
y_Q(1) = 0;
mse_te_k_Q(1) = mean(Desire_test1.^2);
Center(:,1) = Input_train(:,1);
Center_Num = 1;
e_quantz(Center_Num,1,1) = e_k_Q(1);
e_vary(Center_Num,1) = e_k_Q(1);
quantzNum(1) = 1; 
time = zeros(N_tr,1);
% start
for n=2:N_tr
    tic;
    %training
    Kernel = sum((Input_train(:,n)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2,1);
    y_Q(n) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-Kernel./kernelsize))';
    e_tr_Q(n) = Desire_train(n)-y_Q(n);
    [va,index] = min(Kernel);
    if va >= radius_1
         Center_Num = Center_Num+1;
         Center(:,Center_Num) = Input_train(:,n);
         e_k_Q(Center_Num) = e_tr_Q (n);
         e_quantz(Center_Num,1,1) = e_tr_Q(n);
         e_vary(Center_Num,1) = e_tr_Q(n);
         quantzNum(Center_Num) = 1;
     else
         e_k_Q(index) = e_k_Q(index)+e_tr_Q (n);
         quantzNum(index) = quantzNum(index)+1;
         e_quantz(index,quantzNum(index),1) = e_tr_Q(n);
         e_vary(index,quantzNum(index)) = e_k_Q(index);
         e_quantz(index,quantzNum(index),2) = va;
     end
    time(n) = toc;
    %testing MSE
    y_te_Q = zeros(N_te,1);
    if n<(N_tr/2)
    for jj = 1:N_te
        y_te_Q(jj) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-sum((Input_test1(:,jj)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2,1)./kernelsize))';
    end
    err_Q = Desire_test1 -y_te_Q';
    mse_te_k_Q(n) = mean(err_Q.^2);
    else
    for jj = 1:N_te
        y_te_Q(jj) = lr_k_Q*e_k_Q(1:Center_Num)'*(exp(-sum((Input_test2(:,jj)*ones(1,Center_Num)-Center(:,1:Center_Num)).^2,1)./kernelsize))';
    end
    err_Q = Desire_test2 -y_te_Q';
    mse_te_k_Q(n) = mean(err_Q.^2);
    end
    
end

e_changeAll = e_quantz(1:Center_Num,1:max(quantzNum),:);
e_varyAll = e_vary(1:Center_Num,1:max(quantzNum));
end