function [layers] = make_cnn(X_dim, num_class, n_filter, h_filter, w_filter, stride, padding, ~)
conv = Conv(X_dim, n_filter, h_filter, w_filter, 1, padding);
relu_conv = ReLU();
maxpool = Maxpool(conv.out_dim, stride, stride);
flat = Flatten();
drop = Dropout();
fc = FullyConnected(prod(maxpool.out_dim), num_class);
layers = {conv, relu_conv, maxpool, flat, drop, fc};
end

