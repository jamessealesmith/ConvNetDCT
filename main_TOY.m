%% User Interface
close all; clear all; clc;
prepare_workspace();

% Save Settings
will_save = true;
save_string = 'TOY_RESULTS';

% trainer parameters
ntrial = 10;
num_train = 1;
num_test = 1;
verbose = true;

% solver parameters
minibatch_size = 10;
epoch = 10;
learning_rate = 0.1;
mu = 0.9;
solveF = @sgd;

% network parameters
n_filter = 5;
h_filter = 3;
w_filter = 3;
stride = 1;
padding = 1;
s_pool = 1;

% network function
cnnF_list = { @make_cnn @make_cnn_average @make_cnn_fft_lpf @make_cnn_dct_lpf };

% toy parameter
n = 100;


%% End User Interface

% process user interface
[train_set, test_set, data_settings] = load_toy(num_train, num_test,n);
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
        data_mat = [data_mat ; results n_filter h_filter w_filter n];
    catch
        data_mat = [results n_filter h_filter w_filter n];
    end
end



