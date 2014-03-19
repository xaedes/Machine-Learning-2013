function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


Y = eye(num_labels)(y,:);
[J grad] = nnCostFunctionGeneric(nn_params,[input_layer_size,hidden_layer_size,num_labels],X, Y, lambda);
return;

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

% Setup some useful variables
m = size(X, 1);
K = num_labels;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feedforward the neural network

A1 = [ones(size(X,1),1) X]; % Add bias units 

z2 = A1 * Theta1';
A2 = sigmoid(z2); % Compute activation of hidden units

A2 = [ones(size(A2,1),1) A2]; % Add bias units

z3 = A2 * Theta2';
A3 = sigmoid(z3); % Compute activation of output units

% Map y (containing values from 1..K) to a matrix containing binary vectors of 1's and 0's
Y = eye(K)(y,:);

% Compute cost function

h = A3;
J = - sum(sum(Y .* log(h),2)) - sum(sum((1-Y) .* log(1-h),2));

J /= m;

J += (lambda/(2*m)) * (Theta1(:,2:end)(:)' * Theta1(:,2:end)(:));
J += (lambda/(2*m)) * (Theta2(:,2:end)(:)' * Theta2(:,2:end)(:));

% Backpropagation

delta3 = A3 - Y;
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Theta2_grad = delta3' * A2;
Theta1_grad = delta2' * A1;

Theta2_grad /= m;
Theta1_grad /= m;

% size(Theta1_grad)
% size(Theta2_grad)

Theta1_grad(:,2:end) += (lambda / m) * Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda / m) * Theta2(:,2:end);

% [1 0] * [1 0 0]

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
