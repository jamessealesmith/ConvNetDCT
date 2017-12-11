INSTRUCTIONS: Run main file corresponding to the dataset you would like to
   train on. To create a new network, follow the style of the main files.
   Enjoy!

*****************************************************************
  James Smith
  Auburn University
  email  jamessealesmith@auburn.edu
*****************************************************************

NOTE: The structure of many parts of this library are organized in the same
fashion as the PYTHON deepnet libary by parasdahal. Thanks for pointing
me in the right direction for the basic layer and training functions!

https://github.com/parasdahal/deepnet/blob/master/LICENSE

*****************************************************************
main_MNIST: main file for MNIST dataset
main_MNIST_big: main file for modified MNIST dataset
main_TOY: main file for toy dataset
prepare_workspace: adds necessary files to directory

convLayers/
   Conv.m: vanilla convolution layer
   ConvDCT_LPF.m: convolution layer with DCT pooling
   ConvFFT.m: convolution layer utilizing fft
   ConvFFT_LPF.m: convolution layer with FFT pooling
   ConvOA.m: convolution layer utilizing overlapp-and-add

helpers/
   ^conv2olam.m: convolution using overlap-add method
   convfft2.m: 2d convolution utilizing fft2
   get_save_string.m: if desired file location exists, modifies name
   print_results.m: print trainer results
   ^mirt_idctn.m,mirt_dctn.m: DCT using fft implementation

im2col/
   col2im_indices.m
   get_im2col_indices.m
   im2col_indices.m

layers/
   Averagepool.m: average pooling layer
   Batchnorm.m: IN PROGRESS batchnormalization
   Dropout.m: dropout layer
   Flatten.m: flatten layer input
   FullyConnected.m: fully connected MLP layer
   Maxpool.m: max pooling layer
   ReLU.m: rectified linear unit layer
   sigmoid.m: sigmoidal activation layer
   tanh.m: hyberbolic activation layer

loss/
   delta_l1_regularization.m
   delta_l2_regularization.m
   l1_regularization.m
   l2_regularization.m
   softmax_loss.m

make_net/
   make_cnn.m
   make_cnn_average.m
   make_cnn_dct_lpf.m
   make_cnn_dct_lpf.m
   make_cnn_fft.m
   make_cnn_fft_lpf.m
   make_cnn_oa.m

nnet/
   CNN.m

process_data/
   load_mnist.m
   load_mnist_big.m
   load_toy.m
   ^loadMNISTImages.m
   ^loadMNISTLabels.m

training/
   get_minibatches.m
   momentum_update.m
   sgd.m
   sgd_momentum.m
   trainer.m
   vanilla_update.m

utils/
   accuracy
   one_hot_encode
   softmax

visualization/: various scripts to help interpret data

***************************************************************** 
^detonates not written by author
 
OTHER ACKNOWLEDGEMENTS:

MNIST data handling - thanks Stanford UFLDL! 
   http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
MIRT dct2 functions - see license_mirt
OLAM overlap-and-add functions - see license_olam


  Please contribute if you find this software useful.
  Report bugs to jamessealesmith@gmail.com
 