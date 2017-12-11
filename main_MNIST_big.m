%% User Interface
close all; clear all; clc;
prepare_workspace();

% Save Settings
will_save = true;
save_string = 'MNIST_BIG_RESULTS';

% trainer parameters
ntrial = 10;
num_train = 200;
num_test = 50;
verbose = true;

% solver parameters
minibatch_size = 25;
epoch = 30;
learning_rate = 0.0001;
mu = 0.9;
solveF = @sgd_momentum;

% network parameters
n_filter = 32;
h_filter = 3;
w_filter = 3;
stride = 2;
padding = 1;
s_pool = 28;

% network function
cnnF_list = { @make_cnn @make_cnn_average @make_cnn_fft_lpf @make_cnn_dct_lpf };


%% End User Interface

% process user interface
[train_set, test_set, data_settings] = load_mnist_big(num_train, num_test);
train_settings = {ntrial, num_train, num_test, verbose};
solve_settings = {minibatch_size, epoch, learning_rate, mu, solveF};
net_settings = {n_filter, h_filter, w_filter, stride, padding, s_pool};
save_settings = {will_save, save_string};

% loop over networks
for n = 1:length(cnnF_list)
    cnnF = cnnF_list{n};
    % trainer
    fprintf('***** Begin Trainer *****\n\n');
    results = trainer(train_set, test_set, data_settings, train_settings, solve_settings, net_settings, save_settings, cnnF);
    print_results(results);
    fprintf('***** End Trainer *****\n');
    
    % store results
    try
        data_mat = [data_mat ; results n_filter h_filter w_filter -1];
    catch
        data_mat = [results n_filter h_filter w_filter -1];
    end
end


