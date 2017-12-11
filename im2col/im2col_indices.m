function cols = im2col_indices(x, field_height, field_width, padding, stride)

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

x_padded = padarray(x,[0,0,padding,padding],0,'both');

[k, i, j] = get_im2col_indices(size(x), field_height, field_width, padding, stride);

cols = zeros(size(x_padded,1),length(k),length(i));
for a = 1:size(x_padded,1)
    for b = 1:length(k)
        for c = 1:length(i)
            cols(a,b,c) = x_padded(a,k(b)+1,i(b,c)+1,j(b,c)+1);
        end
    end
end
C = size(x,2);
cols = permute(cols,[1,3,2]);
cols = reshape(cols,[],field_height*field_width*C)'; 

end

