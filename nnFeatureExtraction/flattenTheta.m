function [flat] = flattenTheta(Theta)
% typeinfo(Theta)
flat = cell2mat(cellfun(@(m) m(:),Theta,"UniformOutput", false));