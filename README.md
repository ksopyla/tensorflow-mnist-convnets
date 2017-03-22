# Tensorflow MNIST Convolutional Network Tutorial

This project is another tutorial for teaching you Artificial Neural Networks. I hope that my way of presenting the material will help you in long learning process. All the examples are presented in [TensorFlow](https://www.tensorflow.org/) and as a runtime environment I choose the [online python IDE - PLON.io](https://plon.io). PLON makes much easier to share this tutorial with you and run the computations online without any configuraion.

Project presents four different neural nets for [MNIST](http://yann.lecun.com/exdb/mnist/) digit classification. The former two are [fully connected neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) and latter are [convolutional networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). 
Each network is build on top of previous example with gradually increasing difficulty in order to learn more powerful models.



## Models implemented:
* simple single layer neural network (one fully-connected layer) , 
* 5 layer Fully-connected neural network (5 FC NN), 
* convolutional neural netowork: 3x convNet+1FC+output - activation function sigmoid
* convolutional neural netowork with dropout, relu, and better weight initialization: 3x convNet+1FC+output 


## Single layer neural networks

File: minist_single_layer_nn.py

### Network architecture

* Input layer 28*28= 784, 
* Output 10 dim vector (10 digits)

```
input layer             - X[batch, 784]
Fully connected         - W[784,10] + b[10]
One-hot encoded labels  - Y[batch, 10]
```

### Model

```
Y = softmax(X*W+b)
Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]
```

Training consists of finding good W elements. This will be handled automaticaly by Tensorflow optimizer.


## Furhter reading

* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
* [Tensorflow and deep learning without a PHD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/)
* [What is the difference between a Fully-Connected and Convolutional Neural Network?](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/)
