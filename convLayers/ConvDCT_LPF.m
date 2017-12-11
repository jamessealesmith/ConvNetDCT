classdef ConvDCT_LPF < Conv
    
    properties
        s_pool
    end
    
    methods
        function obj = ConvDCT_LPF(X_dim, n_filter, h_filter, w_filter, padding, s_pool)
            obj@Conv(X_dim, n_filter, h_filter, w_filter, 1, padding)
            
            obj.s_pool = s_pool;
            obj.h_out = obj.h_out - 2*s_pool;
            obj.w_out = obj.w_out - 2*s_pool;
            obj.out_dim = [obj.n_filter, obj.h_out, obj.w_out];
        end
        
        function out = forward(obj, X)
            obj.n_X = size(X,1);
                                    
            out = zeros(obj.n_X*obj.d_X,obj.n_filter,obj.h_out,obj.w_out);
            h_c = obj.h_X+obj.h_filter-1;
            w_c = obj.w_X+obj.w_filter-1;           
            for i = 1:obj.n_X
                for j = 1:obj.d_X
                    image_s = squeeze(X(i,j,:,:));
                    for k = 1:obj.n_filter
                        out(i+obj.d_X*(j-1),k,:,:) = obj.dct_forward(obj.conv2olam_forward(...
                            squeeze(obj.params{1}(k,j,:,:)),image_s,...
                            h_c,w_c,obj.h_out+obj.s_pool,obj.w_out+obj.s_pool)...
                            ,obj.h_out,obj.w_out)+obj.params{2}(k);
                    end
                end
            end
            
            
            
        end
        
        function [dX, vout] = backward(obj, dout)
            dout = padarray(dout,[0 0 obj.s_pool obj.s_pool],0,'both'); 
            
            dW = zeros(size(obj.params{1}));
            h_c = obj.h_X+obj.h_filter-1;
            w_c = obj.w_X+obj.w_filter-1; 
            for i = 1:obj.n_X
                for j = 1:obj.d_X
                    dout_s = squeeze(dout(i,j,:,:));
                    for k = 1:obj.n_filter
                        dW(k,j,:,:) = obj.conv2olam_backward(...
                            squeeze(obj.params{1}(k,j,:,:)),dout_s,...
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
        
        function lpf_A = dct_forward(~,A,m,n)
            
            Af = mirt_dctn(A);
            Af = Af(1:m,1:n);
            lpf_A = mirt_idctn(Af);
        end
        
        function conv_mat = conv2olam_forward(~,A,B,h_f,w_f,h,w)
            h_m = floor((h_f - h)) / 2; w_m = ((w_f - w) / 2);
            conv_mat = conv2olam(A,B,0);
            conv_mat = conv_mat(h_m+1:h_m+h,w_m+1:w_m+w);
        end
        
        function conv_mat = conv2olam_backward(~,A,B,h_f,w_f,h,w)
            h_m = floor((h_f - h)) / 2; w_m = ((w_f - w) / 2);
            conv_mat = conv2olam(A,B,0);
            conv_mat = conv_mat(h_m+1:h_m+h,w_m+1:w_m+w);
        end
        
    end
end



