function trainFeaturesNN(X, architecture, filename)
%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

% initial_nn_params = getInitialRandomTheta(architecture);

% size(initial_nn_params)

% load('-binary','Theta.trained.binary');

load('-binary',filename);



displayTrainingFigures(X,Training{1,3})

% % Display some items 
% Theta = Training{1,3};
% size(Theta{1,1})
% p = predictGeneric(Theta,X);

% sel = randperm(size(X, 1));
% sel = sel(1:25);

% figure(2, 'name', 'X');
% displayData(X(sel, :));
% figure(3, 'name', 'h(X)');
% displayData(p(sel, :));
% figure(4, 'name', '|h(X)-X|');
% displayData(abs(p(sel, :)-X(sel, :)));

% figure(5, 'name', '|Theta_1|');
% displayData(Theta{1}(:, 2:end));
% figure(6, 'name', '|Theta_end|');
% displayData(Theta{length(Theta)}(:, 2:end)');

while(true)
    % default values, they should get updated to correct values after one complete for loop below: (but not quite sure if correct)
    bestCost = Training{1,2}; 
    bestTheta = Training{1,3};
    for i=[1]
    % for i=[1:size(Training,1)]

        MaxIter = 50;


        nn_params = flattenTheta(Training{i,3});

        %  You should also try different values of lambda
        lambda = 0;

        %  After you have completed the assignment, change the MaxIter to a larger
        %  value to see how more training helps.
        options = optimset('MaxIter', MaxIter);

        % Create "short hand" for the cost function to be minimized
        costFunction = @(p) nnCostFunctionGeneric(p, ...
                                                  architecture, ...
                                                  X, X, lambda);

        fprintf('id %d - Iteration %d\n', i, Training{i,1});

        % Now, costFunction is a function that takes in only one argument (the
        % neural network parameters)
        [nn_params, cost] = fmincg(costFunction, nn_params, options);
        cost;
        % Obtain Theta back from nn_params
        Theta = getTheta(nn_params, architecture);


        Training{i,1} += MaxIter;
        Training{i,2} = cost;
        Training{i,3} = Theta;

        if(cost<bestCost)
            bestCost = cost;
            bestTheta = Theta;
        end

        displayTrainingFigures(X, Theta);

        % % Display some items 
        % p = predictGeneric(Theta,X);

        % sel = randperm(size(X, 1));
        % sel = sel(1:25);

        % figure(2, 'name', 'X');
        % displayData(X(sel, :));
        % figure(3, 'name', 'h(X)');
        % displayData(p(sel, :));
        % figure(4, 'name', '|h(X)-X|');
        % displayData(abs(p(sel, :)-X(sel, :)));
        % figure(7, 'name', '1-|h(X)-X|');
        % displayData(1-abs(p(sel, :)-X(sel, :)));

        % figure(5, 'name', '|Theta_1|');
        % displayData(Theta{1}(:, 2:end));
        % figure(6, 'name', '|Theta_end|');
        % displayData(Theta{length(Theta)}(:, 2:end)');
 
        save('-binary',filename,'Training');
    end




end

 % displayData(double(reshape(imread("NormalizedBrodatz/D44.tif"),1,640*640))/255.);

 
fprintf('Program paused. Press enter to continue.\n');
pause;