%Copyright
%QZD
%July 4 2008
%
%Description:
%Compare ALD, NC, ENC, KRLS, QKRLS and RFF-RLS using the m-g time series prediction

close all
clc

%% Data Formatting

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

varNoise = 0.001;
inputDimension = 7;

% Data size for training and testing
trainSize = 1000;
testSize = 100;

inputSignal = MK30 + sqrt(varNoise)*randn(size(MK30));
% inputSignal = inputSignal - mean(inputSignal);

%Input training signal with data embedding
trainInput = zeros(inputDimension,trainSize);
for k = 1:trainSize
    trainInput(:,k) = inputSignal(k:k+inputDimension-1);
end

%Input test data with embedding
testInput = zeros(inputDimension,testSize);
for k = 1:testSize
    testInput(:,k) = inputSignal(k+trainSize:k+inputDimension-1+trainSize);
end

% One step ahead prediction
predictionHorizon = 1;

% Desired training signal
trainTarget = zeros(trainSize,1);
for ii=1:trainSize
    trainTarget(ii) = inputSignal(ii+inputDimension+predictionHorizon-1);
end

% Desired training signal
testTarget = zeros(testSize,1);
for ii=1:testSize
    testTarget(ii) = inputSignal(ii+inputDimension+trainSize+predictionHorizon-1);
end


%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 1;
%%
timeNC = zeros(150,1);timeENC = zeros(150,1);
timeALD = zeros(150,1);
timeQ = zeros(150,1);
timeRFF = zeros(150,1);

regularizationFactor = 0.001;
flagLearningCurve = 1;

length_nc = 20;
th_distance_nc_vector = linspace(0.05,0.2,length_nc);
th_error_nc_vector = linspace(0.05,0.2,length_nc);
mse_krls_nc = zeros(length_nc, length_nc);
distsize_krls_nc = zeros(length_nc, length_nc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             KRLS NC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii = 1:length_nc
    for kk = 1:length_nc
        tic;
        th_distance_nc = th_distance_nc_vector(ii);
        th_error_nc = th_error_nc_vector(kk);

        [expansionCoefficient1,dictionaryIndex1,learningCurve1] = ...
            KRLS_NC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,th_distance_nc,th_error_nc,flagLearningCurve);

        y_te = zeros(testSize,1);
        for jj = 1:testSize
            y_te(jj) = expansionCoefficient1*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex1),typeKernel,paramKernel);
        end
        distsize_krls_nc(ii,kk) = length(dictionaryIndex1);
        mse_krls_nc(ii,kk) = mean(learningCurve1(end-99:end));
     timeNC((ii-1)*length_nc+kk) = toc;
    end
end

% =========end of KRLS_NC================




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRLS ALD
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
length_ald = 100;
th_ald_vector = linspace(0.04,0.3,length_ald);
mse_krls_ald = zeros(length_ald, 1);
distsize_krls_ald = zeros(length_ald, 1);
tic;
for ii = 1:length_ald
    tic;
    th_ald = th_ald_vector(ii);
    [expansionCoefficient2,dictionaryIndex2,learningCurve2] = ...
        KRLS_ALD(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,th_ald,flagLearningCurve);
    
    y_te = zeros(testSize,1);
    for jj = 1:testSize
        y_te(jj) = expansionCoefficient2*...
            ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex2),typeKernel,paramKernel);
    end
    distsize_krls_ald(ii) = length(dictionaryIndex2);
    mse_krls_ald(ii) = mean(learningCurve2(end-99:end));
    timeALD(ii) = toc;
end
% =========end of KRLS-ALD================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             KRLS ENC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
length_enc = 20;
th_distance_enc_vector = linspace(0.04,0.2,length_enc);
th_error_enc_vector = linspace(0.05,0.2,length_enc);
mse_krls_enc = zeros(length_enc, length_enc);
distsize_krls_enc = zeros(length_enc, length_enc);
tic;
for ii = 1:length_enc
    for kk = 1:length_enc
        tic;
        th_distance_enc = th_distance_enc_vector(ii);
        th_error_enc = th_error_enc_vector(kk);

        [expansionCoefficient3,dictionaryIndex3,learningCurve3] = ...
            KRLS_ENC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,th_distance_enc,th_error_enc, flagLearningCurve);

        y_te = zeros(testSize,1);
        for jj = 1:testSize
            y_te(jj) = expansionCoefficient3*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex3),typeKernel,paramKernel);
        end
        distsize_krls_enc(ii,kk) = length(dictionaryIndex3);
        mse_krls_enc(ii,kk) = mean(learningCurve3(end-99:end));
     timeENC((ii-1)*length_enc+kk) = toc;
    end
end
% =========end of KRLS_ENC================
ENCtime = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             RFF-RLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pointNum = 100;
steady = zeros(1,pointNum);
MC = 1;
for i = 1:pointNum
    
%close all
%clc
%======filter config=======
%time delay (embedding) length
Dimension = inputDimension;
%data size
N_tr = trainSize;
N_te = testSize;
% first_stage = 1000;
% second_stage = 1300;
mean_learningcurve3=zeros(N_tr,1);
mean_centerQ = 0;
mseMean = 0;
%%======end of config=======

for iteration = 1:MC
    
%=============RLS-RFF-KLMS===================
tic;
D = 20+(i-1)*4;
deltaRFF = 0.707;
forgetFctor = 1;
regularizationFactor = 1;
W = normrnd(0,1/deltaRFF,[D,Dimension])';
learningcurve_RFF = RLS_RFF_KLMS(trainInput,trainTarget,...
    testInput,testTarget,W,D,regularizationFactor,forgetFctor);
mean_learningcurve3 = mean_learningcurve3+learningcurve_RFF;
timeRFF(i) = toc;
%=============end of RLS-RFF-KLMS=================

end

mean_learningcurve3 = mean_learningcurve3./MC;
mean_centerQ = mean_centerQ/MC;
steady(i) = steady(i)+mean(mean_learningcurve3(end-99:end));

end
% =========end of LSRFF================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             QKRLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mean_learningCurve_QKRLS = 0;
regularizationFactor_QKRLS = 0.01;
forgettingFactor = 1;
distance_threshold = 0.16+(0:40)*0.005;
epsilon_ALL = exp(-distance_threshold.^2);
steady_QKRLS = zeros(1,length(epsilon_ALL));
distsize_Qkrls = zeros(1,length(epsilon_ALL));
tic;
for ii = 1:length(epsilon_ALL)
    tic;
        epsilon = epsilon_ALL(ii);
        [expansionCoefficient_QKRLS,dictionaryIndex_QKRLS,learningCurve_QKRLS] = ...
            QKRLS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor_QKRLS,forgettingFactor,epsilon,flagLearningCurve);

        y_te = zeros(testSize,1);
        for jj = 1:testSize
            y_te(jj) = expansionCoefficient_QKRLS*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex_QKRLS),typeKernel,paramKernel);
        end
        distsize_Qkrls(ii) = length(dictionaryIndex_QKRLS);
        %mse_krls_nc(ii,kk) = mean((testTarget - y_te).^2);
        mean_learningCurve_QKRLS = mean_learningCurve_QKRLS+learningCurve_QKRLS;
        steady_QKRLS(ii) = mean(learningCurve_QKRLS(end-99:end));
        timeQ(ii) = toc;
end



figure
hold on
distsize_krls_ncNew = distsize_krls_nc(:);
mse_krls_ncNew = mse_krls_nc(:);
plot(distsize_krls_ncNew,mse_krls_ncNew, '*')
plot(distsize_krls_ald,mse_krls_ald, '+')

distsize_krls_encNew = distsize_krls_enc(:);
mse_krls_encNew = mse_krls_enc(:);
plot(distsize_krls_encNew,mse_krls_encNew, 'd')
plot(distsize_Qkrls,steady_QKRLS,'o');
plot(30+(0:99)*4,steady,'<','LineWidth',1);


set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('network size'),ylabel('testing MSE')

