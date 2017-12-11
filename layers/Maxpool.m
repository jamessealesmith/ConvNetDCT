classdef Maxpool < handle
    
    properties
        params
        d_X
        h_X
        w_X
        size_
        stride
        h_out
        w_out
        out_dim
        max_indexes
        X_col
        n_X
    end
    
    methods
        function obj = Maxpool(X_dim, size_, stride)
            obj.d_X = X_dim(1);
            obj.h_X = X_dim(2);
            obj.w_X = X_dim(3);
            
            obj.params = [];
            
            obj.size_ = size_;
            obj.stride = stride;
            
            obj.h_out = (obj.h_X - size_) / stride + 1;
            obj.w_out = (obj.w_X - size_) / stride + 1;
            
            if(floor(obj.h_out) ~= obj.h_out|| floor(obj.w_out) ~= obj.w_out)
                fprintf("Invalid dimensions!\n")
                return
            end
            
            obj.out_dim = [obj.d_X, obj.h_out, obj.w_out];     
        end
        
        function out = forward(obj, X)
            obj.n_X = size(X,1);
            X_reshaped = reshape(X,[size(X,1)*size(X,2), 1, size(X,3), size(X,4)]);
            
            obj.X_col = im2col_indices(...
                X_reshaped, obj.size_, obj.size_, 0, obj.stride);
            
            [~,obj.max_indexes] = max(obj.X_col,[],1);
            
            out = zeros(1,size(obj.X_col,2));
            for i = 1:length(obj.max_indexes)
                out(i) = obj.X_col(obj.max_indexes(i),i);
            end
            
            out = reshape(out, obj.n_X,obj.d_X, obj.h_out, obj.w_out);
            %out = permute(out,[1,2,4,3]);
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dX_col = zeros(size(obj.X_col));
            
            % flatten gradient
            dout_flat = reshape(permute(dout,[3,4,1,2]),[],1);
            
            
            for i = 1:length(dout_flat)
                dX_col(obj.max_indexes(i),i) = dout_flat(i);
            end
            
            % get original X_reshaped structure from col2im
            shape = [obj.n_X*obj.d_X, 1, obj.h_X, obj.w_X];
            dX = col2im_indices(dX_col, shape, obj.size_, obj.size_, 0, obj.stride); 
            dX = reshape(dX, obj.n_X,obj.d_X, obj.h_X, obj.w_X);
            %dX = permute(dX,[2,1,3,4]);
            vout = [];
        end
        
        
    end
end

