function grads = delta_l1_regularization(layers, grads, lam)
if(nargin < 3)
    lam = 0.001;
end

n_layer = size(layers,1);
for l = 1:n_layer
    layer = layers(l);
    grad = grads(n_layer - l + 1);
    if isprop(layer,'W')
        grad(1) = grad(1) + lam * layer.W / (abs(layer + 1e-8));
    end
end

end

