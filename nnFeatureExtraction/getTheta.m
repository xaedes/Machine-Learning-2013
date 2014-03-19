function Theta = getTheta(nn_params, architecture)

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