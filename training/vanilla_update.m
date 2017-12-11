function vanilla_update(nnet, grads, learning_rate)
if(nargin < 3)
    learning_rate = 0.01;
end
np = length(grads);
for n = 1:np
    for i = 1:length(grads{np-n+1})
        nnet.layers{n}.params{i} = nnet.layers{n}.params{i} - learning_rate * grads{np-n+1}{i};
    end
end
end

