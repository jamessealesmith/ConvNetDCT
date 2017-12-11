classdef ReLU < handle
    
    properties
        params
        X
    end
    
    methods
        function obj = ReLU()
            obj.params = [];
        end
        
        function out = forward(obj, X)
            obj.X = X;
            out = max(X,0);
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX = dout;
            dX(obj.X <= 0) = 0;
            vout = [];
        end
    end
end

