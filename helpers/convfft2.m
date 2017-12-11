function conv_mat = convfft2(A,B)
m = size(A,1)+size(B,1)-1;
n = size(A,2)+size(B,2)-1;
A_f = fft2(A,m,n);
B_f = fft2(B,m,n);
conv_mat_f = A_f.*B_f;
conv_mat = ifft2(conv_mat_f);
end

