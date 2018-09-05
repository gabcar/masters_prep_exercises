from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def example_1():
    # Example 1

    # Create two edges
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0, dtype=tf.float32)

    # Create a node
    total = a + b

    # returns information about the tensors
    print(a)
    print(b)
    print(a + b)

    writer = tf.summary.FileWriter('log/')
    writer.add_graph(tf.get_default_graph())

    # Create a session
    # Calling this session will backtrack all the nodes
    # to the input nodes and run the graph.
    sess = tf.Session()

    # Example: 
    print(sess.run({'ab':(a, b), 'total':total}))

def example_2():
    # Example 2 Randomization
    vec = tf.random_uniform(shape=(3,))
    out1 = vec + 1
    out2 = vec + 2

    print(sess.run(vec))
    print(sess.run(vec))
    print(sess.run((out2, out1)))

def example_3():
    # Example 3 - Operations and placeholders

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y 

    print(sess.run(z, feed_dict={x: 3, y: 4.7}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

def example_4():
    # Example 4 - Datasets

    my_data = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]

    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()


    while True:
        try: 
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break


def example_5():
    # Example 5 - Layers
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    # Initializer initializes the existing layers and no more!
    # call this last.

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6]]}))

def example_6():
    # Example 6 - shortcut functions
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.layers.dense(x, units=1)

    # instantiate initialiser
    init = tf.global_variables_initializer()
    # run initialiser
    sess.run(init)

    print(sess.run(y, feed_dict={x: [[1, 2, 3, 4], 
                                    [5, 6, 7, 8]]}))

def example_7():
    # Example 7 - Feature colums
    sess = tf.Session()

    features = {
        'sales': [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']
    }

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening']
    )
    department_column = tf.feature_column.indicator_column(department_column)
    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess = tf.Session()
    sess.run((var_init, table_init))


    print(sess.run(inputs))

def example_8():

    writer = tf.summary.FileWriter('log/')
    writer.add_graph(tf.get_default_graph())

    # Example 8 - Training a simple mlp
    # Def the data
    x = tf.constant([[1], [2], [3], [4]], 
                    dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], 
                        dtype=tf.float32)

    # construct the graph
    layer_1 = tf.layers.dense(x, units=3)
    layer_2 = tf.layers.dense(layer_1, units=3)
    layer_3 = tf.layers.dense(layer_2, units=3)
    y_pred = tf.layers.dense(layer_3, units=1)

    #init the session and the global variables
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    # Define loss, since regression we have regular 
    # mse
    loss = tf.losses.mean_squared_error(
                        labels=y_true, 
                        predictions=y_pred)

    print(sess.run(loss))

    # Define an optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    print(sess.run(y_pred))
    for _ in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    writer = tf.summary.FileWriter('log/')
    writer.add_graph(tf.get_default_graph())

    print(sess.run(y_pred))



if __name__ == '__main__':
    example_8()