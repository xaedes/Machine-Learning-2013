function [J grad] = nnCostFunctionGeneric(nn_params, ...
                                   architecture, ...
                                   X, Y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a generic
%neural network 
%   [J grad] = NNCOSTFUNCTON(nn_params, architecture, X, y, lambda)
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%
%   architecture is a vector containing the size of each layer (including 
%   input and output layer) 
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the Theta parameters, the weight matrices
% for our neural network

% Setup some useful variables
m = size(X, 1);
num_layers = length(architecture);

Theta = cell(num_layers-1,1);
offset = 1;
for i = 1:(num_layers-1)
    layer_size = architecture(i);
    layer_after_size = architecture(i+1);
    

    Theta{i} = reshape(nn_params(offset:offset + (layer_size+1)*layer_after_size-1), ...
                       layer_after_size, layer_size+1);

    offset += (layer_size+1)*layer_after_size;
end


         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

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

A = cell(num_layers,1);
z = cell(num_layers,1);
A{1} = [ones(size(X,1),1) X];
for i = 2:num_layers
    z{i} = A{i-1} * Theta{i-1}';
    A{i} = sigmoid(z{i});

    if(i < num_layers)
        A{i} = [ones(size(A{i},1),1) A{i}];
    end
end

% Compute cost function

h = A{num_layers};
J = - sum(sum(Y .* log(h),2)) - sum(sum((1-Y) .* log(1-h),2));

J /= m;

% Add regularization
for i = 1:(num_layers-1)
    J += (lambda/(2*m)) * (Theta{i}(:,2:end)(:)' * Theta{i}(:,2:end)(:));

% Backpropagation
delta = cell(num_layers,1);
Theta_grad = cell(num_layers-1,1);

delta{num_layers} = A{num_layers} - Y;
for i=num_layers-1:-1:2
    delta{i} = (delta{i+1} * Theta{i}(:,2:end)) .* sigmoidGradient(z{i});
end

for i=1:num_layers-1
    Theta_grad{i} = delta{i+1}' * A{i};
    Theta_grad{i} /= m;
    Theta_grad{i}(:,2:end) += (lambda / m) * Theta{i}(:,2:end);  % Add regularization
end


% [1 0] * [1 0 0]

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = cell2mat(cellfun(@(m) m(:),Theta_grad,"UniformOutput", false));

end
