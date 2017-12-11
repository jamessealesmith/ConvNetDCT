function onehot = one_hot_encode(y, num_class)
m = y.size(0);
onehot = zeros(m,num_class);
for i = 1:m
    idx = y(i);
    onehot(i,idx) = 1;
end

