classdef sigmoid < handle
    
    properties
        out
    end
    
    methods
        function obj = sigmoid()
        end
        
        function out = forward(obj, X)
            out = 1 / (1 + exp(X));
            obj.out = out;
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX = dout * obj.out * (1 - obj.out);
            vout = [];
            
        end
    end
end

