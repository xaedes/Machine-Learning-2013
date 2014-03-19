function nnFeatureExtraction()

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

% figure(1);
% displayData(X(sel, :));



fprintf('Program paused. Press enter to continue.\n');
% pause;

%% Set up network architecture
num_input = size(X,2);

% number of units per hidden layer (except features layer)
hidden = floor(num_input * 1);

% layer containing features of input as activation
features = 10;

% Layers for extraction of features and reconstruction of input from features
extraction = [hidden; hidden];
reconstruction = [hidden; hidden];

% architecture = [num_input; extraction; features; reconstruction; num_input]

architecture = [num_input; 40 ;20 ; 40; num_input]
% architecture = [num_input; hidden; 200; hidden; num_input]

% return;

trainFeaturesNN(X,architecture,'Training.binary');

 % displayData(double(reshape(imread("NormalizedBrodatz/D44.tif"),1,640*640))/255.);

 
fprintf('Program paused. Press enter to continue.\n');
pause;