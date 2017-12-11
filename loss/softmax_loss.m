function [loss, dx] = softmax_loss(X, y)
m = size(y,1);
p = softmax(X);
log_likelihood = -log(p(y+1));
loss = sum(log_likelihood) / m;

dx = p;
for i = 1:m
    dx(i,y(i)+1) = dx(i,y(i)+1) - 1;
end
dx = dx / m;
end

