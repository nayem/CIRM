from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math

import tensorflow as tf
import scipy.io as sio
import numpy as np

def weight_variable (shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def next_batch(batch_size):
    return [np.random.randint(195192, size=(batch_size,10) ), np.random.randint(195192, size=(batch_size,5) ) ]



X = tf.placeholder( tf.float32, shape=[None, 10])
Y = tf.placeholder( tf.float32, shape=[None, 5])

weights = {
    'h1': weight_variable([10, 10]),
    'out': weight_variable([10, 5])
}

new_w = dict()

layer_1 = tf.matmul(X, weights['h1'])
out_layer = tf.matmul(layer_1, weights['out'])

loss = tf.reduce_mean(tf.square(Y - out_layer))
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

    batch_x, batch_y = next_batch(30)

    _, c, w= sess.run([train_op, loss, weights], feed_dict={X: batch_x, Y: batch_y})
    new_w = w.copy()

    w1, b1 = [new_w['h1'], new_w['h1']]
    print(w1)
    print(b1)