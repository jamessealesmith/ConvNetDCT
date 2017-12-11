function [layers] = make_cnn_average(X_dim, num_class, n_filter, h_filter, w_filter, stride, padding, ~)
conv = Conv(X_dim, n_filter, h_filter, w_filter, 1, padding);
relu_conv = ReLU();
avepool = Averagepool(conv.out_dim, stride, stride);
flat = Flatten();
drop = Dropout();
fc = FullyConnected(prod(avepool.out_dim), num_class);
layers = {conv, relu_conv, avepool, flat, drop, fc};
end

