# encoding: UTF-8
# Copyright Krzysztof Sopyła (krzysztofsopyla@gmail.com)
#
#
# Licensed under the MIT

# Network architecture:
# Five layer neural network, input layer 28*28= 784, output 10 (10 digits)
# Output labels uses one-hot encoding

# input layer             - X[batch, 784]
# 1 layer                 - W1[784, 200] + b1[200]
#                           Y1[batch, 200] 
# 2 layer                 - W2[200, 100] + b2[100]
#                           Y2[batch, 200] 
# 3 layer                 - W3[100, 60]  + b3[60]
#                           Y3[batch, 200] 
# 4 layer                 - W4[60, 30]   + b4[30]
#                           Y4[batch, 30] 
# 5 layer                 - W5[30, 10]   + b5[10]
# One-hot encoded labels    Y5[batch, 10]

# model
# Y = softmax(X*W+b)
# Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

# Training consists of finding good W elements. This will be handled automaticaly by 
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
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# layers sizes
L1 = 200
L2 = 100
L3 = 60
L4 = 30
L5 = 10

# weights - initialized with random values from normal distribution mean=0, stddev=0.1
# output of one layer is input for the next
W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
b1 = tf.Variable(tf.zeros([L1]))

W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b2 = tf.Variable(tf.zeros([L2]))

W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.zeros([L3]))

W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.zeros([L4]))

W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
b5 = tf.Variable(tf.zeros([L5]))



# flatten the images, unrole eacha image row by row, create vector[784] 
# -1 in the shape definition means compute automatically the size of this dimension
XX = tf.reshape(X, [-1, 784])

# Define model
Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
Y1 = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y2 = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y3 = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Y4 = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
#cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 100.0  # normalized for batches of 100 images,


# we can also use tensorflow function for softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

                                                          
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, 
learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)


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
            # compute training values for visualisation
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
            
            
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_trn)
            test_acc.append(acc_tst)

        # the backpropagationn training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75})


title = "MNIST 5 layer, relu, adam, dropout"
vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,DISPLAY_STEP)



# Restults
# mnist_single_layer_nn.py acc= 0.9237 
# mnist__layer_nn.py TST acc = 0.9534
# mnist__layer_nn_relu_adam.py TST acc = 0.9771
# mnist__layer_nn_relu_adam_dropout.py TST acc = 0.9732

# sample output for 5k iterations
#0 Trn acc=0.10000000149011612 , Trn loss=229.3443603515625 Tst acc=0.11999999731779099 , Tst loss=230.12518310546875
#100 Trn acc=0.9300000071525574 , Trn loss=30.25579071044922 Tst acc=0.8877000212669373 , Tst loss=35.22196578979492
#200 Trn acc=0.8799999952316284 , Trn loss=33.183040618896484 Tst acc=0.9417999982833862 , Tst loss=19.18865966796875
#300 Trn acc=0.9399999976158142 , Trn loss=21.5306396484375 Tst acc=0.9406999945640564 , Tst loss=19.576183319091797
# ...
#4800 Trn acc=0.9700000286102295 , Trn loss=11.897256851196289 Tst acc=0.9749000072479248 , Tst loss=9.952529907226562
#4900 Trn acc=0.9800000190734863 , Trn loss=4.757292747497559 Tst acc=0.974399983882904 , Tst loss=11.507346153259277
#5000 Trn acc=0.9900000095367432 , Trn loss=4.661561012268066 Tst acc=0.9732999801635742 , Tst loss=11.199274063110352
