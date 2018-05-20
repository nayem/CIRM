from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math

import tensorflow as tf
import scipy.io as sio
import numpy as np


def weight_variable(shape, stddev=None, stddev_2=None):
    if stddev is None:
        initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / sum(shape)))
    elif stddev > 0.0:
        if stddev_2 is not None:
            r, c = shape
            initial1 = tf.truncated_normal([r, c//2], stddev=stddev)
            initial2 = tf.truncated_normal([r, c//2], stddev=stddev_2)
            initial = tf.concat([initial1,initial2], axis=1)
        else:
            initial = tf.truncated_normal(shape, stddev=stddev)

    else:
        initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)


def bias_variable(shape, stddev=None, stddev_2=None):
    # initial = tf.constant(0.1, shape=shape)
    if stddev is None:
        initial = tf.truncated_normal(shape, stddev=np.sqrt(1.0 / sum(shape)))
    elif stddev > 0.0:
        if stddev_2 is not None:
            r, c = shape
            initial1 = tf.truncated_normal([r, c//2], stddev=stddev)
            initial2 = tf.truncated_normal([r, c//2], stddev=stddev_2)
            initial = tf.concat([initial1,initial2], axis=1)
        else:
            initial = tf.truncated_normal(shape, stddev=stddev)

    else:
        initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)

c = 6
r = 2

X = tf.constant([[1.,2.,3.],[4.,5.,6.]])
Y = tf.constant([[0.,1.,2.,4.,2.,5.],[1.,2.,3.,4.,5.,6.]])

L2 = tf.get_variable("my_variable", [4, 4])

# Model parameters
W = weight_variable([3,6],0.001,0)
b = bias_variable([1,6],0.4, 0)



diag_1 = tf.diag(weight_variable([3],0.001))
diag_2 = tf.diag(weight_variable([3],0.001))
diag_3 = tf.diag(weight_variable([3],0.001))
diag_4 = tf.diag(weight_variable([3],0.001))

d_out = tf.concat( [tf.concat([diag_1, diag_2], 1), tf.concat([diag_3, diag_4], 1)],0)
b2 = bias_variable([1,6],0.4, 0)


layer_1 = tf.nn.relu(tf.matmul(X, W) + b)
predictions = tf.nn.relu(tf.matmul(layer_1, d_out) + b2 )

cost1 = tf.reduce_sum(tf.squared_difference(Y[:, :c//2], predictions[:, :c//2]))
cost2 = tf.reduce_sum(tf.squared_difference(Y[:, c//2:], predictions[:, c//2:]))
loss_t = 0.5*(cost1+cost2)/r


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
d, d_1, d_2, c1,c2,loss = sess.run([d_out, diag_1,diag_2, cost1, cost2, loss_t])

print('d',d.shape)
# print(d)

print('d1',d_1.shape)
# print(d_1)

print('d2',d_2.shape)
# print(d_2)

print('c1:',c1, 'c2:', c2, 'loss:',loss)