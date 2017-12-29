# encoding: UTF-8
# Copyright Krzysztof Sopyła (krzysztofsopyla@gmail.com)
#
#
# Licensed under the MIT

# Network architecture:
# Single layer neural network, input layer 28*28= 784, output 10 (10 digits)
# Output labels uses one-hot encoding

# input layer             - X[batch, 784]
# Fully connected         - W[784,10] + b[10]
# One-hot encoded labels  - Y[batch, 10]

# model
# Y = softmax(X*W+b)
# Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

# Training consists of finding good W elements. This will be handled automatically by 
# Tensorflow optimizer


import visualizations as vis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


NUM_ITERS=5000
DISPLAY_STEP=100
BATCH=100

tf.set_random_seed(0)

# Download images and labels 
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

# mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
# mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

# Placeholder for input images, each data sample is 28x28 grayscale images
# All the data will be stored in X - tensor, 4 dimensional matrix
# The first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10] - initialized with random values from normal distribution mean=0, stddev=0.1
W = tf.Variable(tf.truncated_normal([784, 10],stddev=0.1))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images, unroll each image row by row, create vector[784] 
# -1 in the shape definition means compute automatically the size of this dimension
XX = tf.reshape(X, [-1, 784])

# Define model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualization
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])


# Initializing the variables
init = tf.global_variables_initializer()

train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)


    for i in range(NUM_ITERS+1):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(BATCH)

        if i%DISPLAY_STEP ==0:
            # compute training values for visualization
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
                        
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        # the back-propagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

    

title = "MNIST 1.0 single softmax layer"
vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,DISPLAY_STEP)


# sample output for 5k iteration 
# one layer TST acc= 0.9237

#0 Trn acc=0.10000000149011612 , Trn loss=257.5960998535156 Tst acc=0.09359999746084213 , Tst loss=262.1988525390625
#100 Trn acc=0.8799999952316284 , Trn loss=42.37456512451172 Tst acc=0.879800021648407 , Tst loss=41.94294357299805
#200 Trn acc=0.8299999833106995 , Trn loss=50.943721771240234 Tst acc=0.883899986743927 , Tst loss=39.74076843261719
#300 Trn acc=0.8500000238418579 , Trn loss=39.26817321777344 Tst acc=0.9021999835968018 , Tst loss=33.9247932434082
# ....
#4800 Trn acc=0.9100000262260437 , Trn loss=32.18285369873047 Tst acc=0.9223999977111816 , Tst loss=27.490428924560547
#4900 Trn acc=0.9399999976158142 , Trn loss=19.267147064208984 Tst acc=0.9240999817848206 , Tst loss=27.271032333374023
#5000 Trn acc=0.949999988079071 , Trn loss=14.251474380493164 Tst acc=0.923799991607666 , Tst loss=27.61037826538086