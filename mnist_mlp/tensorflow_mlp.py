from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor as tpe

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
Step 0: Read the data
"""
mnist = input_data.read_data_sets('../data/', one_hot=True)

"""
Step 1: Define all the parameters
"""
lr = 0.001                  # irrelevant for adam
n_epochs = 1                # Number of epochs
batch_size = 100            # batch size

print(mnist.train.images.shape)

n_features = 28 * 28        # mnist images are 28x28 but flattened
n_classes = 10              # 10 digits. 

hidden_dims = [64,          # Dimensions of hidden layers
               32,
               16]

"""
Step 2: Pre-define the network structure and the input format with placeholders and 
layers.
"""
X = tf.placeholder('float', [None, n_features])
Y = tf.placeholder('float', [None, n_classes])

def mlp_model(x):
    _x = x
    for dims in hidden_dims:
        _x = tf.layers.dense(_x, units=dims)
    out = tf.layers.dense(_x, units=n_classes)
    return out


logits = mlp_model(X)

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
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        # Number of batches for the whole dataset
        n_batches = mnist.train.num_examples // batch_size

        for i in range(n_batches):
            # Extract the batch from the training set.
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization on training loss
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
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
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    writer = tf.summary.FileWriter('log/')
    writer.add_graph(tf.get_default_graph())

