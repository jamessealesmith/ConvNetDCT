function reg_loss = l2_regularization(layers, lam)
if(nargin < 2)
    lam = 0.001;
end
reg_loss = 0.0;
nl = length(layers);
for n = 1:nl
    if(length(layers{n}.params) > 1)
        reg_loss = reg_loss + 0.5 * lam * sum(sum(sum(sum(abs(layers{n}.params{1}.*layers{n}.params{1})))));
    end
end
end

