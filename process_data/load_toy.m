function [train_set, test_set, data_settings] = load_toy(num_training, num_test,n)
if(nargin < 1)
    num_training = 3;
end
if(nargin < 2)
    num_test = 3;
end
% Old toy for debugging
% num_class = 3;

% prepare training data
% X_train = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0;1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0];
% X_train = reshape(X_train,[],1,5,5);
% X_train = permute(X_train,[1 2 4 3]);
% y_train = [0;1;2];

% prepare testing data
% X_test = X_train;
% y_test = y_train;

num_class = 10;
X_train = rand(num_training,1,n,n);
y_train = randi(num_class-1,num_training,1);

X_test = rand(num_test,1,n,n);
y_test = randi(num_class-1,num_test,1);

train_set = {X_train, y_train};
test_set = {X_test, y_test};

data_dim = size(X_train);
data_dim = data_dim(2:4);
data_settings = {data_dim, num_class};
end

