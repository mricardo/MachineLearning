function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

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

C_candidates     = []; # [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_candidates = []; # [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

best_pred = -1;
for i = 1:length(C_candidates),
  for j = 1:length(sigma_candidates),
    model = svmTrain(X, y, C_candidates(i), @(x1, x2) gaussianKernel(x1, x2, sigma_candidates(j))); 
    
    predictions = svmPredict(model, Xval);
    pred = mean(double(predictions ~= yval));
    if (pred < best_pred || best_pred == -1)
      best_pred = pred;
      C = C_candidates(i);
      sigma = sigma_candidates(j);
    endif
  endfor
endfor

C = 1;
sigma = 0.1;
disp("C: "); disp(C);
disp("Sigma: "); disp(sigma);
      
end
