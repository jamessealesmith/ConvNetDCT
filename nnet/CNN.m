classdef CNN < handle
    
    properties
        layers
        loss_func
    end
    
    methods
        function obj = CNN(layers, loss_func)
            if(nargin < 2)
                loss_func = @softmax_loss;
            end
            obj.layers = layers;
            obj.loss_func = loss_func;
        end
        
        function X = forward(obj, X)
            for l = 1:length(obj.layers)
                X = obj.layers{l}.forward(X);
            end
        end
        
        function grads = backward(obj, dout)
            grads = {};
            for l = length(obj.layers):-1:1
                [dout, grad] = obj.layers{l}.backward(dout);
                grads = [grads {grad}];
            end
        end
        
        function [loss, grads] = train_step(obj, X, y)
           out = obj.forward(X);
           [loss, dout] = obj.loss_func(out, y);
           loss = loss + l2_regularization(obj.layers);
           grads = obj.backward(dout);
           grads = delta_l2_regularization(obj.layers, grads);
        end
        
        function prediction = predict(obj, X)
           X = obj.forward(X);
           [~,prediction] = max(softmax(X),[], 2);
           prediction = prediction - 1;
        end
    end
end

