classdef tanh < handle
    
    properties
        out
    end
    
    methods
        function obj = tanh()
        end
        
        function out = forward(obj, X)
            out = tanh(X);
            obj.out = out;
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX = dout * (1 - obj.out.^2);
            vout = [];
        end
    end
end

