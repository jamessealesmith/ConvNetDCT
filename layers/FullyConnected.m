classdef FullyConnected < handle
    
    properties
        params
        X
    end
    
    methods
        function obj = FullyConnected(in_size, out_size)
            W = rand(in_size, out_size) / sqrt(in_size/2);
            %W = ones(in_size,out_size);
            b = zeros(1,out_size);
            obj.params = {W, b};
        end
        
        function out = forward(obj, X)
            obj.X = X;
            out = obj.X * obj.params{1} + obj.params{2};
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dW = obj.X' * dout;
            db = sum(dout,1);
            dX = dout * obj.params{1}';
            vout = {dW, db};
            
        end
    end
end

