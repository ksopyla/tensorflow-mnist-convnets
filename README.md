# Tensorflow MNIST Convolutional Network Tutorial

This project is another tutorial for teaching you Artificial Neural Networks. I hope that my way of presenting the material will help you in long learning process. All the examples are presented in [TensorFlow](https://www.tensorflow.org/) and as a runtime environment I choose the [online python IDE - PLON.io](https://plon.io). PLON makes much easier to share this tutorial with you and run the computations online without any configuraion.

Project presents four different neural nets for [MNIST](http://yann.lecun.com/exdb/mnist/) digit classification. The former two are [fully connected neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) and latter are [convolutional networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). 
Each network is build on top of previous example with gradually increasing difficulty in order to learn more powerful models.


## Models implemented

* simple single layer neural network (one fully-connected layer) , 
* 5 layer Fully-connected neural network (5 FC NN) in 3 variants
* convolutional neural netowork: 3x convNet+1FC+output - activation function sigmoid
* convolutional neural netowork with dropout, relu, and better weight initialization: 3x convNet+1FC+output 





## Single layer neural network

This is the simplest architecture that we will consider. This neural network will be our baseline model for further more powerfull solutions.
This is also a good candidate for understanding Tensorflow execution model.

File: **minist\_1.0\_single\_layer\_nn.py**

### Network architecture

* Input layer 28*28= 784, 
* Output 10 dim vector (10 digits, one-hot encoding)

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

## Five layers fully-connected neural network

This is upgraded version of previous model, between input and output we added five fully connected hidden layers. Adding more layers makes network more expressive but in the same time harder to train. The three new problems could emerge: vanising gradients, model overfitting and computation time complexity. In our case where the dataset is rather small, we did not see those problems in real scale.

In order to deal with those problems, different training techniques was invented. Changeing from sigmoid to relu activation function will prevent vanising gradients, chosing Adam optimizer will speed up  optimization and in the same time shorten training time, adding dropout will help with overfitting.

This model was implemented in three variants:

* five layer fully-connected 
* five layer fully-connected with relu activation function and Adam optmizer
* five layer fully-connected with relu activation, Adam optmizer and dropout

## Network architecture

```
input layer             - X[batch, 784]
1 layer                 - W1[784, 200] + b1[200]
                          Y1[batch, 200] 
2 layer                 - W2[200, 100] + b2[100]
                          Y2[batch, 200] 
3 layer                 - W3[100, 60]  + b3[60]
                          Y3[batch, 200] 
4 layer                 - W4[60, 30]   + b4[30]
                          Y4[batch, 30] 
5 layer                 - W5[30, 10]   + b5[10]
One-hot encoded labels    Y5[batch, 10]

model
Y = softmax(X*W+b)
Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]
```
 

## Refernces and furhter reading

* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
* [Tensorflow and deep learning without a PHD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/) - very good tutorial showing how to build modern MNIST conv net. It was my inspiration for this tutorial :)
* [What is the difference between a Fully-Connected and Convolutional Neural Network?](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/)
* [Tensorflow Examples by aymericdamien](aymericdamien/TensorFlow-Examples) - github repository with very useful and not so obious Tensorflow examples
* [Awesome tensorflow](https://github.com/jtoy/awesome-tensorflow) - A curated list of dedicated resources

* [Projects with #Tensorflow tag in plon.io](https://plon.io/explore/tag/tensoflow)