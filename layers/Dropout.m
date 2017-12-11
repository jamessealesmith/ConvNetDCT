classdef Dropout < handle
    
    properties
        params
        prob
        mask
    end
    
    methods
        function obj = Dropout(varargin)
            if(nargin < 1)
                obj.prob = 0.5;
            else
                obj.prob = vargin;
            end
            obj.params = [];
        end
        
        function out = forward(obj, X)
            obj.mask = binornd(1, obj.prob, size(X)) / (1 - obj.prob);
            out = X.* obj.mask;
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX = dout .* obj.mask;
            vout = [];
        end
    end
end

