function accuracy = svmAccuracy(model,X,y)

pred = svmPredict(model,X);

accuracy = sum(pred == y) / length(y);