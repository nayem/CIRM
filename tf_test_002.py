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


# Model parameters
W = weight_variable([3,4],0.001,0)
b = bias_variable([1,4],0.4, 0)
r = 100
c = 4

# mse = tf.losses.mean_squared_error(labels=W, predictions=b,weights=0.5)
# mse2 = tf.losses.mean_squared_error(labels=W, predictions=b)
# loss_t = tf.divide(tf.reduce_sum(mse), r)

# mse_r = tf.losses.mean_squared_error(labels=W[:,:c//2], predictions=b[:,:c//2])
# mse_i = tf.losses.mean_squared_error(labels=W[:, c//2:], predictions=b[:, c//2:])
# loss_d = tf.divide( (-tf.reduce_sum(mse_r)-tf.reduce_sum(mse_i)), r)

p = tf.constant([[4,5]])
r, c = p.get_shape().as_list()
print(type(r), r, type(c), c)


X = tf.constant([[1.,2.,3.],[4.,5.,6.]])
Y = tf.constant([[0.,1.,2.],[1.,2.,3.]])
m = tf.reduce_sum(tf.squared_difference(X, Y), axis=0)
k = 0.5*Y/2.0

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('mse: ', sess.run([m,k]))

# w,b=sess.run([W, b])
# w,b,m,m2,l=sess.run([W, b, mse_r, mse_i, loss_d])
# print('w:\n',w)
# print('b:\n',b)
# print('m:\n',m)
# print('m2:\n',m2)
# print('l:\n',l)