function displayTrainingFigures(X, Theta, figurestart = 2)
    % Display some items 
    p = predictGeneric(Theta,X);

    sel = randperm(size(X, 1));
    sel = sel(1:25);

    i = figurestart, figure(i, 'name', 'X');
    displayData(X(sel, :));
    i += 1; figure(i, 'name', 'h(X)');
    displayData(p(sel, :));
    i += 1; figure(i, 'name', '|h(X)-X|');
    displayData(abs(p(sel, :)-X(sel, :)));
    i += 1; figure(i, 'name', '1-|h(X)-X|');
    displayData(1-abs(p(sel, :)-X(sel, :)));

    i += 1; figure(i, 'name', '|Theta_1|');
    displayData(Theta{1}(:, 2:end));
    i += 1; figure(i, 'name', '|Theta_end|');
    displayData(Theta{length(Theta)}(:, 2:end)');
end