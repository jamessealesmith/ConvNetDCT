function [k, i, j] = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

if(nargin < 2)
    field_height = 3;
end
if(nargin < 3)
    field_width = 3;
end
if(nargin < 4)
    padding = 1;
end
if(nargin < 5)
    stride = 1;
end

N = x_shape(1);
C = x_shape(2);
H = x_shape(3);
W = x_shape(4);
assert(mod(H + 2*padding - field_height, stride) == 0);
assert(mod(W + 2*padding - field_width, stride) == 0);
out_height = (H + 2 * padding - field_height) / stride + 1;
out_width = (W + 2 * padding - field_width) / stride + 1;

i0 = repmat(0:field_height-1,field_width,1);
i0 = reshape(i0,[1 field_width*field_height]);
i0 = repmat(i0,1,C);
i1 = repmat(0:out_height-1,out_width,1);
i1 = stride * reshape(i1,[1 out_width*(out_height)]);
j0 = repmat(0:field_width-1,1,field_height*C);
j1 = stride * repmat(0:out_width-1,1,out_height);
i = i0' + i1;
j = j0' + j1;

k = repmat(0:C-1,field_height*field_width,1);

end
