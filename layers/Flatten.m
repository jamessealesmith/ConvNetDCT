classdef Flatten < handle
    
    properties
        params
        X_shape
        out_shape
    end
    
    methods
        function obj = Flatten()
            obj.params = [];
            
        end
        
        function out = forward(obj, X)
            obj.X_shape = size(X);
            out = reshape(permute(X,[1 4 2 3]),size(X,1),[]);
            obj.out_shape = -1;
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX = reshape(dout,obj.X_shape);
            vout = [];
        end
    end
end

