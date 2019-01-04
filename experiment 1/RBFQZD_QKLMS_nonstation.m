%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright QZD 
%IAIR
%2015-12-24 11:16:10
%
%description:
%compare the performance of kernelsize and error iteration QKLMS
%for Mackey Glass time series
%one step prediction
%Learning curves
%ʵ�鷢�ֺ˿���0.5-1��ʱ�������Ӱ�컹��ͦ���Եģ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
%close all
%clc
%======filter config=======
%time delay (embedding) length
Dimension = 1;
%kernel parameter
a = 1;%fixed
%noise std
np = sqrt(0.5);
%data size
N_tr = 600;
N_te = 100;
% first_stage = 1000;
% second_stage = 1300;
mean_learningcurve1=zeros(N_tr,1);
mean_learningcurve2=zeros(N_tr,1);
mean_learningcurve3=zeros(N_tr,1);
mean_centerQ = 0;
mseMean = 0;
time1 = 0;
time2 = 0;
time3 = 0;
%%======end of config=======
MC = 50;

for iteration = 1:MC
    
 %fprintf('Learning curves are generating. Please wait! MC = %f \n',iteration);   
%======data formatting===========
input = rand(1,1000);
output = zeros(size(input));
output(1:500) = sin(10*input(1:500));
output(501:1000) = sin(12*input(501:1000));
Input_train = input(201:800);
noise = normrnd(0,sqrt(0.5),[1,600]);
Desire_train = output(201:800)+noise;
Input_test1 = input(1:100);
Desire_test1 = output(1:100);
Input_test2 = input(801:900);
Desire_test2 = output(801:900);

%======end of data formatting===========


%=============RFF-KLMS===================
tic;
D = 10;
deltaRFF = 0.1;
lr_RFF = 0.01;
W = normrnd(0,1/deltaRFF,[D,Dimension])';
[learningcurve_RFF,time_RFF] = RFF_KLMS(Input_train,Desire_train...
    ,Input_test1,Desire_test1,Input_test2,Desire_test2,W,D,lr_RFF);
mean_learningcurve1 = mean_learningcurve1+learningcurve_RFF;
time1=time1+time_RFF;
%=============end of RFF-KLMS=================


%=============QLMS===================
%learning rate (adjustable)
%lr_k = .1;
%lr_k_Q = 1.2;
lr_k_Q = .1;
radius_1 = 0;
initialdelta = .1;
%e_Q������centerϵ�� ,  e_quantz������u�����e������u��center�ľ���
%e_vary��ÿ��centerϵ���仯����,  quantzNum��ÿ��center����u�ĸ���
[learningcurve,CenterQ,Center_Num_Q,e_Q,e_quantz,e_vary,quantzNum,time] =...
    QKLMS(Input_train,Desire_train,Input_test1,Desire_test1,Input_test2,Desire_test2,lr_k_Q,radius_1,initialdelta);
mean_learningcurve2 = mean_learningcurve2+learningcurve;
mean_centerQ = mean_centerQ+Center_Num_Q;
time2 = time2+time;
%=========end of KLMS================

%=============RLS-RFF-KLMS===================
tic;
D = 10;
deltaRFF = 0.1;
forgetFctor = 0.99;
regularizationFactor = 1.5;
W = normrnd(0,1/deltaRFF,[D,Dimension])';
[learningcurve_RFF,time_RLS] = RLS_RFF_KLMS(Input_train,Desire_train...
    ,Input_test1,Desire_test1,Input_test2,Desire_test2,W,D,regularizationFactor,forgetFctor);
mean_learningcurve3 = mean_learningcurve3+learningcurve_RFF;
time3 = time3+time_RLS;
%=============end of RLS-RFF-KLMS=================

end
mseMean = mseMean./MC;
mean_learningcurve1 = mean_learningcurve1./MC;
mean_learningcurve2 = mean_learningcurve2./MC;
mean_learningcurve3 = mean_learningcurve3./MC;
mean_centerQ = mean_centerQ/MC;
time1 = time1./MC;
time2 = time2./MC;
time3 = time3./MC;

figure
% plot(mean_learningcurve,'r','LineWidth',2);
% hold on
plot(mean_learningcurve1,'b','LineWidth',2);
hold on
plot(mean_learningcurve2,'g','LineWidth',2);
hold on
plot(mean_learningcurve3,'r','LineWidth',2);
hold on

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('RFF-KLMS','KLMS','RFF-RLS')
xlabel('iteration')
ylabel('MSE')
set(gca, 'YScale','log')

figure; 
plot(time1,'LineWidth',2);
hold on
plot(time2,'LineWidth',2);
plot(time3,'LineWidth',2);
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('RFF-KLMS','KLMS','RFF-RLS')
xlabel('iteration')
ylabel('CPU time on one iteration')
% disp('>>EFF-KLMS                 cNum    R   delta stpsize')
% mseMean = mean(mean_learningcurve1(end-99:end,1));
% mseStd = std(mean_learningcurve1(end-99:end,1));
% disp([num2str(mseMean),'+/-',num2str(mseStd),' | ',num2str(mean_centerQ),' | ',num2str(radius_1),' | ',num2str(initialdelta),' | ',num2str(lr_k_Q)]);
% disp('>>QKLMS                  cNum    R   delta stpsize')
% mseMean = mean(mean_learningcurve2(end-99:end,1));
% mseStd = std(mean_learningcurve2(end-99:end,1));
% disp([num2str(mseMean),'+/-',num2str(mseStd),' | ',num2str(mean_centerQ),' | ',num2str(radius_1),' | ',num2str(initialdelta),' | ',num2str(lr_k_Q)]);
% disp('>>RLS-EFF-KLMS              cNum    R   delta stpsize')
% mseMean = mean(mean_learningcurve3(end-99:end,1));
% mseStd = std(mean_learningcurve3(end-99:end,1));
% disp([num2str(mseMean),'+/-',num2str(mseStd),' | ',num2str(mean_centerQ),' | ',num2str(radius_1),' | ',num2str(initialdelta),' | ',num2str(lr_k_Q)]);
