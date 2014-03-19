function initialTheta = getInitialRandomTheta(architecture)


num_layers = length(architecture);

Theta = cell(num_layers-1,1);
offset = 1;
for i = 1:(num_layers-1)
    layer_size = architecture(i);
    layer_after_size = architecture(i+1);
    
    Theta{i} = randInitializeWeights(layer_after_size, layer_size);
end

initialTheta = Theta;