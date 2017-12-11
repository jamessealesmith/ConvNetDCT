function [minibatchesX, minibatchesy] = get_minibatches(X, y, minibatch_size, shufflev)
if(nargin < 4)
    shufflev = true;
end
m = size(X,1);
minibatchesX = {};
minibatchesy = {};
if(shufflev)
    ind = randperm(m);
    X = X(ind,:,:,:);
    y = y(ind,:);
end
dimN = length(size(X));
for i = 1:minibatch_size:m
    try
        X_batch = X(i:i+minibatch_size-1,:,:,:);
        y_batch = y(i:i+minibatch_size-1);
    catch
        X_batch = X(i:end,:,:,:);
        y_batch = y(i:end);
    end
    minibatchesX = [minibatchesX, X_batch];
    minibatchesy = [minibatchesy, y_batch];
end
end

