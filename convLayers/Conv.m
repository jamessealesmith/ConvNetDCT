classdef Conv < handle
    
    properties
        params
        d_X
        h_X
        w_X
        n_filter
        h_filter
        w_filter
        stride
        padding
        h_out
        w_out
        out_dim
        n_X
        X_col
    end
    
    methods
        function obj = Conv(X_dim, n_filter, h_filter, w_filter, stride, padding)
            obj.d_X = X_dim(1);
            obj.h_X = X_dim(2);
            obj.w_X = X_dim(3);
            obj.n_filter = n_filter;
            obj.h_filter = h_filter;
            obj.w_filter = w_filter;
            obj.stride = stride;
            obj.padding = padding;
            
            W = rand(n_filter, obj.d_X, h_filter, w_filter) / sqrt(n_filter/2);
            %W = ones(n_filter, obj.d_X, h_filter, w_filter);
            b = zeros(obj.n_filter,1);
            obj.params = {W, b};
            
            obj.h_out = (obj.h_X - h_filter + 2*padding) / stride + 1;
            obj.w_out = (obj.w_X - w_filter + 2*padding) / stride + 1;
            
            assert(floor(obj.h_out) == obj.h_out);
            assert(floor(obj.w_out) == obj.w_out);
            
            obj.out_dim = [obj.n_filter, obj.h_out, obj.w_out];
        end
        
        function out = forward(obj, X)
            obj.n_X = size(X,1);
            
            obj.X_col = im2col_indices(...
                X, obj.h_filter, obj.w_filter, obj.stride, obj.padding);
            W_row = reshape(obj.params{1},obj.n_filter,[]);
            
            out = W_row * obj.X_col + obj.params{2};
            out = reshape(out, obj.n_filter,[], obj.h_out, obj.w_out);
            out = permute(out, [2,1,4,3]);           
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dout_flat = obj.shape_d_out(dout);
            
            dW = dout_flat * obj.X_col';
            dW = reshape(dW,size(obj.params{1}));
            dW = permute(dW,[1,2,4,3]);
            
            db = sum(sum(sum(dout,1),3),4);
            db = reshape(db, obj.n_filter,[]);
            
            W_flat = reshape(obj.params{1}, obj.n_filter, []);
            
            dX_col = W_flat'*dout_flat;
            shape = [obj.n_X, obj.d_X, obj.h_X, obj.w_X];
            dX = col2im_indices(dX_col, shape, obj.h_filter, obj.w_filter, obj.padding, obj.stride);
            
            vout = {dW, db};
        end
        
        function dout_flat = shape_d_out(~,dout)
            dout_flat = zeros(size(dout,2),numel(dout)/size(dout,2));
            for i2 = 1:size(dout,2)
                a = 1;
                for i3 = 1:size(dout,3)
                    for i4 = 1:size(dout,4)
                        for i1 = 1:size(dout,1)
                            dout_flat(i2,a) = dout(i1,i2,i3,i4);
                            a = a+1;
                        end
                    end
                end
            end
        end
        
    end
end

