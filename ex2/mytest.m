
function X,y = mytest(z)

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

% [X, mu, sigma] = featureNormalize(X);

X = [ones(size(X,1), 1) X];

theta = zeros(size(X,2),1);
% theta = 0.1*ones(size(X,2),1)

[J grad] = costFunctionReg([1;0;0;-1], [1 1 1 1; 1 0 1 0; 1 1 0 0; 1 1 1 1], [0;1;1;0], .5)