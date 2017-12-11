function [nnet, loss, train_acc, test_acc, save_out] = sgd_momentum(nnet, X_train, y_train, minibatch_size,...
    epoch, learning_rate, mu, verbose, will_save, X_test, y_test, nesterov)
if(nargin <12)
    nesterov = true;
end
if(will_save)
    loss_save = zeros(1,epoch);
    train_acc_save = zeros(1,epoch);
    test_acc_save = zeros(1,epoch);
end

[minibatchesX,minibatchesY] = get_minibatches(X_train, y_train, minibatch_size,false);
minibatch_size = length(minibatchesX);
n_layer = size(nnet.layers,2);
for i = 1:epoch
    loss = 0;
    velocity = {};
    for l = 1:n_layer
        plz = {};
        for pl = 1:length(nnet.layers{l}.params)
            plz = {plz{:}; zeros(size(nnet.layers{l}.params{pl}))};
        end
        velocity = [velocity {plz}];
    end
    if verbose
        fprintf('Epoch %d',i);
    end
    for b = 1:minibatch_size
        X_mini = minibatchesX{b};
        y_mini = minibatchesY{b};
        
        if nesterov
            for l = 1:n_layer
                ve = velocity{l};
                for il = 1:length(nnet.layers{l}.params)
                    nnet.layers{l}.params{il} = nnet.layers{l}.params{il} + mu * ve{il};
                end
            end
        end
        
        [loss, grads] = nnet.train_step(X_mini, y_mini);
        [velocity] = momentum_update(velocity,nnet, grads, learning_rate, mu);
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

