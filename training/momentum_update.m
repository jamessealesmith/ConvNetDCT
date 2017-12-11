function [velocity] = momentum_update(velocity, nnet, grads, learning_rate, mu)
if(nargin < 3)
    learning_rate = 0.01;
end
if(nargin < 4)
    mu = 0.9;
end
np = length(grads);
for n = 1:np
    for i = 1:length(grads{np-n+1})
        velocity{n}{i} = mu * velocity{n}{i} + learning_rate * grads{np-n+1}{i};
        nnet.layers{n}.params{i} = nnet.layers{n}.params{i} - velocity{n}{i};
    end
end
end

