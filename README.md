# Tensorflow MNIST Convolutional Network Tutorial

Project presents different neural nets for MNIST digit classification. There are 4 models implemented, 
two fully connected neural networks and two convolutional networks.

Models:
* simple single layer NN, 
* 5 layer FC NN 
* 3xconvNet+1FC+output - activation function sigmoid
* 3xconvNet+1FC+output - Activation function 

## Single layer neural networks

File: minist_single_layer_nn.py


### Network architecture:
Single layer neural network, input layer 28*28= 784, output 10 (10 digits)
Output labels uses one-hot encoding
input layer             - X[batch, 784]
Fully connected         - W[784,10] + b[10]
One-hot encoded labels  - Y[batch, 10]
model
Y = softmax(X*W+b)
Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]
Training consists of finding good W elements. This will be handled automaticaly by Tensorflow optimizer
