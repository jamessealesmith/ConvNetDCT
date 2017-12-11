classdef Batchnorm < handle
    
    properties
        params
        d_X
        h_X
        w_X
        gamma
        beta
        n_X
        X_shape
        X_flat
        mu
        var_
        X_norm
    end
    
    methods
        function obj = Batchnorm(X_dim)
            obj.d_X = X_dim(1);
            obj.h_X = X_dim(2);
            obj.w_X = X_dim(3);
            gamma = ones(1,obj.d_X*obj.h_X*obj.w_X);
            beta = zeros(1,obj.d_X*obj.h_X*obj.w_X);
            obj.params = [gamma, beta];
        end
        
        function out = forward(obj, X)
            obj.n_X = size(X,1);
            obj.X_shape = size(X);
            obj.X_flat = reshape(reshape(X,1),obj.n_X,[]);
            obj.mu = mean(obj.X_flat,1);
            obj.var_ = var(obj.X_flat,1);
            obj.X_norm = (obj.X_flat - obj.mu) / sart(obj.var_ + 1e-8);
            out = obj.params{1} * obj.X_norm + obj.params{2};
            out = reshape(out,obj.X_shape);
        end
        
        function [dX, vout] = backward(obj, dout)
            dout = reshape(reshape(dout,1),size(dout,1),[]);
            X_mu = obj.X_flat - obj.mu;
            var_inv = 1 / sqrt(obj.var_ + 1e-8);
            
            dbeta = sum(dout,1);
            dgamma = dout * obj.X_norm;
            
            dX_norm = dout * obj.params{1};
            dvar = sum(dX_norm * X_mu,1) * - 0.5 * (obj.var_ + 1e-8).^(-3/2);
            dmu = sum(dX_norm * -var_inv,1) + dvar * 1 / obj.n_X * sum(-2*X_mu,0);
            dX = (dX_norm * var_inv) + (dmu / sobj.n_X) + (dvar * 2 / obj.n_X * X_mu);
            
            dX = reshape(dX,obj.X_shape);
            vout = [dgamma, dbeta];
        end
    end
end

