function p = predictGeneric(Theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
num_layers = length(Theta)+1;

A = cell(num_layers,1);
A{1} = [ones(size(X,1),1) X];
for i = 2:num_layers
    A{i} = sigmoid(A{i-1} * Theta{i-1}');

    if(i < num_layers)
        A{i} = [ones(size(A{i},1),1) A{i}];
    end
end


p = A{num_layers};
% =========================================================================


end
