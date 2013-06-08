function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

allC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
allSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
bestC = allC(1);
bestSigma = allSigma(1);
previousErr = 1000;
for i=1:length(allC)
    currentC = allC(i);
    for j=1:length(allSigma)
        currentSigma = allSigma(j);
        model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        if err < previousErr
            bestC = currentC;
            bestSigma = currentSigma;
            previousErr = err;
        end;
    end;
end;

C = bestC;
sigma = bestSigma;

% =========================================================================

end
