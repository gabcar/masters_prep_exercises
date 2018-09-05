"""tensorflow_lenet.py
Tensorflow implementation of LeNet MNIST classifier from
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor as tpe

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


"""
Step 0: Read the data
"""
mnist = input_data.read_data_sets('../data/', 
                                  one_hot=True,
                                  reshape=False)

train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
print(test_labels.shape)

# Pad to lazily convert images to 32x32 for lenet...
train_data = np.pad(train_data, ((0, 0),
                                 (2, 2),
                                 (2, 2),
                                 (0, 0)),
                                 'constant')
test_data = np.pad(test_data, ((0, 0),
                               (2, 2),
                               (2, 2),
                               (0, 0)),
                               'constant')

"""
Step 1: Define all the parameters
"""
lr = 0.001                  # irrelevant for adam
n_epochs = 10                # Number of epochs
batch_size = 100            # batch size

rows, cols = (32, 32)
n_classes = 10              # 10 digits. 

X = tf.placeholder(tf.float32, [None, rows, cols, 1])
Y = tf.placeholder(tf.int32, [None, n_classes])
one_hot_Y = tf.placeholder(tf.int32, [None, n_classes])

"""
Step 2: Pre-define the network structure and the input format with placeholders and 
layers.
"""

def lenet_model(x):
    conv_1 = tf.layers.conv2d(inputs=x, 
                              filters=6, 
                              kernel_size=5, 
                              activation=tf.nn.relu)
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, 
                                     pool_size=2, 
                                     strides=2)
    
    conv_2 = tf.layers.conv2d(inputs=pool_1, 
                              filters=16, 
                              kernel_size=5, 
                              activation=tf.nn.relu)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, 
                                     pool_size=2, 
                                     strides=2)

    flat = tf.layers.flatten(pool_2)

    linear_1 = tf.layers.dense(flat, 
                               120, 
                               activation=tf.nn.relu)
    linear_2 = tf.layers.dense(linear_1,
                               84,
                               activation=tf.nn.relu)
    
    return tf.layers.dense(linear_2, 10)


logits = lenet_model(X)

"""
Step 3: Define loss function, optimizer and the initialiser
object.
"""

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

""" 
Step 4: Initialize the variables and commence training. 
"""
# We run session with context manager for KeyboardInterrupt saves
# helpful when logging with tensorboard.
with tf.Session() as sess:
    sess.run(init)
    
    n_batches = mnist.train.num_examples // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0
        # Number of batches for the whole dataset

        for i in range(n_batches):
            # Have to resize the batches to meet torch specifications
            
            train_batch = train_data[i * batch_size : (i + 1) * batch_size,
                                        :, :, :]
            train_labels_batch = train_labels[i * batch_size : (i + 1) * batch_size,
                                                :]
            _, loss = sess.run([train_op, loss_op], feed_dict={X: train_batch,
                                            Y: train_labels_batch})
            epoch_loss += loss
        # compute the average loss for all batches. 
        epoch_loss /= n_batches

        print("Epoch {} - Average batch loss: {}".format(epoch + 1, epoch_loss))

    """
    Step 5: evaluate the model
    """

    predictions = tf.nn.softmax(logits)  
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test_data, Y: test_labels}))