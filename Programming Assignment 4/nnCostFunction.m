function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
   a1 = [1 ; X(i, :)'];
   z2 = Theta1 * a1;
   
   a2 = [1; sigmoid(z2)];
   
   z3 = Theta2 * a2;
   a3 = hx = sigmoid(z3);
   
   yk = zeros(num_labels, 1);
   yk(y(i)) = 1;
   
   K = sum(-yk .* log(hx) - (1 - yk) .* log(1 - hx));
   
   J += K;
   
   sigma3 = a3 - yk;
   sigma2 = Theta2' * sigma3 .* sigmoidGradient([1; z2]);
   sigma2 = sigma2(2:end);
   
   Theta1_grad = Theta1_grad + sigma2 * a1';
   Theta2_grad = Theta2_grad + sigma3 * a2';
endfor

reg1 = (lambda * Theta1) / m;
reg2 = (lambda * Theta2) / m;

reg1(:, 1) = 0;
reg2(:, 1) = 0;

Theta1_grad = Theta1_grad(:) / m + reg1(:);
Theta2_grad = Theta2_grad(:) / m + reg2(:);

grad = [Theta1_grad ; Theta2_grad];

regularization = sum([Theta1(:, 2:end)(:); Theta2(:, 2:end)(:)] .^ 2);
regularization = (lambda * regularization) / (2 * m);

J = (J/m) + regularization;

end
