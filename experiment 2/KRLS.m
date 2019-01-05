% function [expansionCoefficient,dictionaryIndex,learningCurve,networkSizeCurve, CI] = ...
%     KRLS(trainInput,trainTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor,th1)
function [expansionCoefficient,dictionaryIndex,learningCurve,networkSizeCurve] = ...
    KRLS(trainInput,trainTarget,typeKernel,paramKernel,regularizationFactor,forgettingFactor)
% modified by Hong: the original KRLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KRLS_ALDs Kernel Recursive Least Squares with approximate linear
%dependency
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%testInput:     testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:    desired signal for testing testSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%regularizationFactor: regularization parameter in Newton's recursion
%
%forgettingFactor: expoentially weighted value
%
%th1:           thresholds used in approximate linear dependency
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%learningCurve:     trainSize*1 used for learning curve
%networkSizeCurve:  trainSize*1 used for network growth curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% memeory initialization
[inputDimension,trainSize] = size(trainInput);

learningCurve = zeros(trainSize,1);
learningCurve(1) = trainTarget(1)^2;


Q_matrix = 1/(forgettingFactor*regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));

expansionCoefficient = Q_matrix*trainTarget(1);
% dictionary
dictionaryIndex = 1;
dictSize = 1;

networkSizeCurve = zeros(trainSize,1);
networkSizeCurve(1) = 1;

predictionVar = regularizationFactor*forgettingFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel);

% CI = zeros(trainSize,1);
% 
% CI(1) = log(predictionVar)/2;

% start training
for n = 2:trainSize
    
    %calc the Conditional Information
    k_vector = ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    networkOutput = expansionCoefficient*k_vector;
    predictionError = trainTarget(n) - networkOutput;
    f_vector = Q_matrix*k_vector;
    
    predictionVar = regularizationFactor*forgettingFactor^(n) + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) -...
        k_vector'*f_vector;
	
%     CI(n) = log(predictionVar)/2;
%     
%     if (CI(n) > th1)

        %update Q_matrix
        s = 1/predictionVar;
        Q_tmp = zeros(dictSize+1,dictSize+1);
        Q_tmp(1:dictSize,1:dictSize) = Q_matrix + f_vector*f_vector'*s;
        Q_tmp(1:dictSize,dictSize+1) = -f_vector*s;
        Q_tmp(dictSize+1,1:dictSize) = Q_tmp(1:dictSize,dictSize+1)';
        Q_tmp(dictSize+1,dictSize+1) = s;
        Q_matrix = Q_tmp;

        % updating coefficients
        dictSize = dictSize + 1;
        dictionaryIndex(dictSize) = n;
        expansionCoefficient(dictSize) = s*predictionError;
        expansionCoefficient(1:dictSize-1) = expansionCoefficient(1:dictSize-1) - f_vector'*expansionCoefficient(dictSize);

        learningCurve(n) = predictionError^2;
        networkSizeCurve(n) =  networkSizeCurve(n-1)+1;
    
%     else  %redundant
%        learningCurve(n) = learningCurve(n-1);
%        networkSizeCurve(n) =  networkSizeCurve(n-1);
%   
%     end
end

return