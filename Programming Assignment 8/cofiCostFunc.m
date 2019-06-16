function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

J = sum(sum(R .* ((X * Theta' - Y) .^ 2))) / 2 + ...
    (lambda/2) * (sum(sum(X.^2)))  + ...
    (lambda/2) * (sum(sum(Theta.^2)));

% one for looping over movies to compute X_grad for each movie
% X_grad = 1682x10
for i = 1:num_movies
  % all users that have rated movie 'i'
  idx = find(R(i, :) == 1);
  
  % only the theta that users have made a rate to movie 'i'
  
  Theta_t = Theta(idx, :);
  
  % only the ratings of users that have rated
  Y_t = Y(i, idx);
  
  X_grad(i, :) = (X(i, :) * Theta_t' - Y_t) * Theta_t + lambda*X(i,:);
endfor

% one for looping over users to compute Theta_Grad for each user
% Theta_grad = 944x10
for j = 1:num_users
  % all movies that user 'j' has rated
  idx = find(R'(j, :));
  
  % only the movies that user 'j' has rated
  X_t = X(idx, :);
  
  % only the movies user 'j' has rated
  Y_t = Y'(j, idx);
  
  Theta_grad(j, :) = (Theta(j, :) * X_t' - Y_t) * X_t + lambda * Theta(j, :);
endfor

grad = [X_grad(:); Theta_grad(:)];

end
