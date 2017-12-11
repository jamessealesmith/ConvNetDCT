function [layers] = make_cnn_fft_lpf(X_dim, num_class, n_filter, h_filter, w_filter, ~, padding, s_pool)
conv = ConvFFT_LPF(X_dim, n_filter, h_filter, w_filter, padding, s_pool);
relu_conv = ReLU();
flat = Flatten();
drop = Dropout();
fc = FullyConnected(prod(conv.out_dim), num_class);
layers = {conv, relu_conv, flat, drop, fc};
end

