function [train_set, test_set, data_settings] = load_mnist_big(num_training, num_test)
if(nargin < 1)
    num_training = 600;
end
if(nargin < 2)
    num_test = 100;
end
num_class = 10;

% citation: http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
images_train = loadMNISTImages('train-images-idx3-ubyte');
labels_train = loadMNISTLabels('train-labels-idx1-ubyte');
images_test = loadMNISTImages('t10k-images-idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

% prepare training data
X_train_s = images_train(1:num_training,:,:,:);
y_train = labels_train(1:num_training);

% prepare testing data
X_test_s = images_test(1:num_test,:,:,:);
y_test = labels_test(1:num_test);

m = size(X_train_s,3);
n = size(X_train_s,4);
X_train = zeros(num_training,size(X_train_s,2),4*m,4*n);
for i = 1:4
    for j = 1:4
        X_train(:,:,(i-1)*m+1:i*m,(j-1)*n+1:j*n) = X_train_s;
    end
end

m = size(X_test_s,3);
n = size(X_test_s,4);
X_test = zeros(num_test,size(X_test_s,2),4*m,4*n);
for i = 1:4
    for j = 1:4
        X_test(:,:,(i-1)*m+1:i*m,(j-1)*n+1:j*n) = X_test_s;
    end
end

train_set = {X_train, y_train};
test_set = {X_test, y_test};

data_dim = size(X_train);
data_dim = data_dim(2:4);
data_settings = {data_dim, num_class};
end

