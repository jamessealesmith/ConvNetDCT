function out = softmax(x)
exp_x = exp(x - max(x,[],2));
out = exp_x ./ sum(exp_x,2);
end

