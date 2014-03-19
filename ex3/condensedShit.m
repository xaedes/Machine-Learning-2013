load('ex3data1.mat'); % training data stored in arrays X, y
X = [ones(m, 1) X]; % Add ones to the X data matrix
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1); % You need to return the following variables correctly 
function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples
n = length(theta); % number of features
h = sigmoid(X * theta);
J =  (-(1 / m) * (y' * log(h) + (1-y)' * log(1 - h)) )  + (lambda/(2*m))*theta(2:n)'*theta(2:n);
grad = (1/m) * (X'*(h-y));
grad(2:n) = grad(2:n) + (lambda/m)*theta(2:n);
end
for c = 1:num_labels
    initial_theta = zeros(n, 1); % Set Initial theta
    options = optimset('GradObj', 'on', 'MaxIter', 50); % Set options for fminunc
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), 0.1)), initial_theta, options);
    all_theta(c,:) = theta';
end
[all_theta] = oneVsAll(X, y, 10, 0.1);