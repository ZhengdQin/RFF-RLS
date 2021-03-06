function [expansionCoefficient,dictionaryIndex,learningCurve] = ...
    KRLS_NC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,toleranceDistance,tolerancePredictError,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KRLS
%kernel recursive least squares using novelty criterion
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
%toleranceDistance:         tolerance for the closeness of the new data to the dictionary
%tolerancePredictError:     tolerance for the apriori error
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.


% memeory initialization
[inputDimension,trainSize] = size(trainInput);
testSize = length(testTarget);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

Q_matrix = 1/(regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));

expansionCoefficient = Q_matrix*trainTarget(1);
% dictionary
dictionaryIndex = 1;
dictSize = 1;

predictionVar = regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel);
predictionError = trainTarget(1);

toleranceDistanceSq = toleranceDistance^2;

% start training
for n = 2:trainSize
    
    % comparing the distance between trainInput(:,n) and the dictionary
    distance2dictionary = min(sum((trainInput(:,n)*ones(1,dictSize) - trainInput(:,dictionaryIndex)).^2));
    if (distance2dictionary < toleranceDistanceSq)
        if flagLearningCurve, learningCurve(n) = learningCurve(n-1); end
        continue;
    end
   
    k_vector = ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    networkOutput = expansionCoefficient*k_vector;
    predictionError = trainTarget(n) - networkOutput;
    
    if (abs(predictionError) < tolerancePredictError)
        if flagLearningCurve==1, learningCurve(n) = learningCurve(n-1); end
        continue;
    end
    
    f_vector = Q_matrix*k_vector;
    
    predictionVar = regularizationFactor + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) -...
        k_vector'*f_vector;
        
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

    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
             y_te(jj) = expansionCoefficient*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end

end

return