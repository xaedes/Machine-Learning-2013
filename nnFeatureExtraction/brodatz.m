
img = double(imread('NormalizedBrodatz/D20.tif'))/255.;
% img = double(imread('NormalizedBrodatz/D5.tif'))/255.;
figure(1,'name','Picture');
displayData(reshape(img,1,640*640));

[xx,yy] = meshgrid(0:(640/32)-2);

xx = 1+reshape(xx * 32, 1,(640/32-1)**2);
yy = 1+reshape(yy * 32, 1,(640/32-1)**2);


blocks = double(cell2mat(arrayfun(@(id) reshape(img(yy(id):64+yy(id)-1,xx(id):64+xx(id)-1),64*64,1), 1:(640/32-1)**2, 'UniformOutput', false)))';

blocks = blocks(1:50,:);

% typeinfo(blocks)
% displayData(double(blocks)/255.);

%% Set up network architecture
num_input = size(blocks,2);

% number of units per hidden layer (except features layer)
hidden = floor(num_input * 1);

% layer containing features of input as activation
features = 10;

% Layers for extraction of features and reconstruction of input from features
extraction = [hidden; hidden];
reconstruction = [hidden; hidden];

% architecture = [num_input; extraction; features; reconstruction; num_input]

% architecture = [num_input; 25; num_input]
architecture = [num_input; 25; 25; 25; num_input]

Training = createRandomTraining(architecture,5);
filename = 'brodatz.D20.Training.binary';
save('-binary',filename,'Training');

trainFeaturesNN(blocks,architecture,filename);

% displayData()