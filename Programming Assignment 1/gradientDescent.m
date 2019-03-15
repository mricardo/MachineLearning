function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

for iter = 1:num_iters    
    J_history(iter) = computeCost(X, y, theta);
    
    h = X * theta;        
    delta = sum((h - y) .* X) / m;
    theta = theta - alpha * delta';   
end

end
