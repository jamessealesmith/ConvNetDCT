function [nnet, loss, train_acc, test_acc, save_out] = sgd(nnet, X_train, y_train, minibatch_size,...
    epoch, learning_rate, ~, verbose, will_save, X_test, y_test)
if(will_save)
    loss_save = zeros(1,epoch);
    train_acc_save = zeros(1,epoch);
    test_acc_save = zeros(1,epoch);
end

[minibatchesX,minibatchesY] = get_minibatches(X_train, y_train, minibatch_size,false);
minibatch_size = length(minibatchesX);
for i = 1:epoch
    loss = 0;
    if verbose
        fprintf('Epoch %d',i);
    end
    for b = 1:minibatch_size
        X_mini = minibatchesX{b};
        y_mini = minibatchesY{b};

        [loss, grads] = nnet.train_step(X_mini, y_mini);
        vanilla_update(nnet, grads, learning_rate);
    end
    train_acc = accuracy(y_train, nnet.predict(X_train));
    test_acc = accuracy(y_test, nnet.predict(X_test));
    if (verbose)
        fprintf(' - Loss = %.4f | Train = %.4f | Test = %.4f\n',...
            loss, train_acc, test_acc);
    end
    if(will_save)
        loss_save(i) = loss;
        train_acc_save(i) = train_acc;
        test_acc_save(i) = test_acc;
    end
end

if(will_save)
    save_out = {loss_save, train_acc_save, test_acc_save};
else
    save_out = {};
end

end

