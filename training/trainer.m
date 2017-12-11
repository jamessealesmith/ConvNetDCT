function [results] = trainer(train_set, test_set, data_settings,...
    train_settings, solve_settings, net_settings, save_settings, cnnF)
% Trainer Parameters
ntrial = train_settings{1};
num_training = train_settings{2};
num_test = train_settings{3};
verbose = train_settings{4};

% Solver Parameters
minibatch_size = solve_settings{1};
epoch = solve_settings{2};
learning_rate = solve_settings{3};
mu = solve_settings{4};
solveF = solve_settings{5};

% Network Parameters
n_filter = net_settings{1};
h_filter = net_settings{2};
w_filter = net_settings{3};
stride = net_settings{4};
padding = net_settings{5};
s_pool = net_settings{6};

% Save Settings
will_save = save_settings{1};
save_string = get_save_string(save_settings{2});

% Process Data
X_train = train_set{1};
y_train = train_set{2};
X_test = test_set{1};
y_test = test_set{2};
data_dim = data_settings{1};
num_class = data_settings{2};

if(size(X_train,2) > 1)
    fprintf('NEED TO CHECK SOME RESHAPE PROBLEMS IN LAYERS FOR 3 COLOR PROBLEMS!!!');
    pause();
end

if(will_save)
    loss_save = zeros(ntrial,epoch);
    train_acc_save = zeros(ntrial,epoch);
    test_acc_save = zeros(ntrial,epoch);
end
results = zeros(1,4);
for tr = 1:ntrial
    cnn = CNN(cnnF(data_dim, num_class, n_filter, h_filter, w_filter,...
        stride, padding, s_pool));
    tic
    [cnn, loss, train_acc, test_acc, save_out] = solveF(cnn, X_train, y_train,...
        minibatch_size, epoch, learning_rate, mu, verbose, will_save, X_test, y_test);
    stop_time = toc;
    results = (results * (tr-1) + [loss train_acc test_acc stop_time]) / tr;
    fprintf('Trial %d | Training Time = %.2f | ', tr, stop_time);
    fprintf('Loss = %.4f | Training Accuracy = %.4f | Testing Accuracy = %.4f\n',...
        loss, train_acc, test_acc);
    
    if(will_save)
        loss_save(tr,:) = save_out{1};
        train_acc_save(tr,:) = save_out{2};
        test_acc_save(tr,:) = save_out{3};
    end
end
if(will_save)
    save(save_string,'loss_save','train_acc_save','test_acc_save','net_settings','cnnF','results');
end
    
end

