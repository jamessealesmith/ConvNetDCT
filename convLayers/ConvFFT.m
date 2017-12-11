classdef ConvFFT < Conv
    
    properties

    end
    
    methods
        function obj = ConvFFT(X_dim, n_filter, h_filter, w_filter, padding)
            
            obj@Conv(X_dim, n_filter, h_filter, w_filter, 1, padding)

        end
        
        function out = forward(obj, X)
            obj.n_X = size(X,1);
                                    
            out = zeros(obj.n_X*obj.d_X,obj.n_filter,obj.h_out,obj.w_out);
            h_c = obj.h_X+obj.h_filter-1;
            w_c = obj.w_X+obj.w_filter-1;           
            for i = 1:obj.n_X
                for j = 1:obj.d_X
                    image_f = fft2(squeeze(X(i,j,:,:)),h_c,w_c);
                    for k = 1:obj.n_filter
                        out(i+obj.d_X*(j-1),k,:,:) = obj.convfft2_forward(...
                            squeeze(obj.params{1}(k,j,:,:)),image_f,...
                            h_c,w_c,obj.h_out,obj.w_out)+obj.params{2}(k);
                    end
                end
            end
        end
        
        function [dX, vout] = backward(obj, dout)
                      
            dW = zeros(size(obj.params{1}));
            h_c = obj.h_X+obj.h_filter-1;
            w_c = obj.w_X+obj.w_filter-1; 
            for i = 1:obj.n_X
                for j = 1:obj.d_X
                    dout_f = fft2(squeeze(dout(i,j,:,:)),h_c,w_c);
                    for k = 1:obj.n_filter
                        dW(k,j,:,:) = obj.convfft2_backward(...
                            squeeze(obj.params{1}(k,j,:,:)),dout_f,...
                            h_c,w_c,obj.h_filter,obj.w_filter);
                    end
                end
            end
                       
            db = sum(sum(sum(dout,1),3),4);
            db = reshape(db, obj.n_filter,[]);
            
            W_flat = reshape(obj.params{1}, obj.n_filter, []);
            dout_flat = obj.shape_d_out(dout);
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
        
        function conv_mat = convfft2_forward(~,A,B_f,h_f,w_f,h,w)
            A_f = fft2(A,h_f,w_f);
            conv_mat_f = A_f.*B_f;
            h_m = floor((h_f - h)) / 2; w_m = ((w_f - w) / 2);
            conv_mat = ifft2(conv_mat_f);
            conv_mat = conv_mat(h_m+1:h_m+h,w_m+1:w_m+w);
        end
        
        function conv_mat = convfft2_backward(~,A,B_f,h_f,w_f,h,w)
            A_f = fft2(A,h_f,w_f);
            conv_mat_f = A_f.*B_f;
            h_m = floor((h_f - h)) / 2; w_m = ((w_f - w) / 2);
            conv_mat = ifft2(conv_mat_f);
            conv_mat = conv_mat(h_m+1:h_m+h,w_m+1:w_m+w);
        end
        
    end
end

