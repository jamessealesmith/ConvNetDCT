function x_padded = col2im_indices(cols, x_shape, field_height, field_width, padding, stride)

if(nargin < 3)
    field_height = 3;
end
if(nargin < 4)
    field_width = 3;
end
if(nargin < 5)
    padding = 1;
end
if(nargin < 6)
    stride = 1;
end

N = x_shape(1);
C = x_shape(2);
H = x_shape(3);
W = x_shape(4);

H_padded = H + 2*padding;
W_padded = W + 2*padding;

x_padded = zeros(N,C,H_padded,W_padded);
[k, i, j] = get_im2col_indices(x_shape, field_height, field_width, padding, stride);


cols_reshaped = reshape(cols,C*field_height*field_width,N,[]);
cols_reshaped = permute(cols_reshaped,[2,1,3]);

for a = 1:size(x_padded,1)
    for b = 1:length(k)
        for c = 1:length(i)
            x_padded(a,k(b)+1,i(b,c)+1,j(b,c)+1) = x_padded(a,k(b)+1,i(b,c)+1,j(b,c)+1) + cols_reshaped(a,b,c);
        end
    end
end

if(padding ~= 0)
    x_padded = x_padded(:,:,padding+1:end-padding,padding+1:end-padding);
end
end

