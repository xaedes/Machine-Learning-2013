function Training = createRandomTraining(architecture, num)

Training = cell(num,3);
for i=1:size(Training,1)
    Training{i,1} = 0;
    Training{i,2} = 0;
    Training{i,3} = getInitialRandomTheta(architecture);
end