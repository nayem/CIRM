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

# para
ns = 5
c = 6
prpi = np.array( [[1,2,3,9,7,8] , [4, 5, 6,1,2,3], [7, 8,9,5,6,4], [0, 1, 4,0, 1, 0],[ 4, 8, 9,4,2,1]]  )
one = np.ones((5,3))
zero = np.zeros((5,3))
lrli = np.concatenate( (one, zero), axis=1)

# pr = [1,2,3 ; 4, 5, 6; 7, 8,9; 0, 1, 4; 4, 8, 9]
# pi = [9,7,8 ; 1,2,3; 5,6,4; 0, 1, 0; 4,2,1]
# lr = ones(5,3)
# li = zeros(5,3)



# Cost Function
y = tf.placeholder(tf.float32, shape=[None, 6])
predictions = tf.placeholder(tf.float32, shape=[None, 6])

cost1 = tf.reduce_sum(tf.squared_difference(y[:, :c//2], predictions[:, :c//2]))
cost2 = tf.reduce_sum(tf.squared_difference(y[:, c//2:], predictions[:, c//2:]))
loss_t = 0.5*(cost1+cost2)/ns

mse_r = tf.reduce_sum(tf.squared_difference(y[:, :c // 2], predictions[:, :c // 2]), axis=0)
mse_i = tf.reduce_sum(tf.squared_difference(y[:, c // 2:], predictions[:, c // 2:]), axis=0)
loss_d = -tf.reduce_mean(mse_r)-tf.reduce_mean(mse_i)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# print('mse: ', sess.run([m,k]))

c1, c2, l_t, mr, mi, l_d = sess.run([cost1,cost2,loss_t ,mse_r,mse_i,loss_d], feed_dict={y:lrli, predictions:prpi})
print('Train:', c1, c2, l_t)
print ('Dev: ',mr, mi, l_d )

sum_mse_r = np.zeros((3))
sum_mse_i = np.zeros((3))

sum_mse_r += (mr)
sum_mse_i += (mi)

avg_cost = - np.mean(sum_mse_r) - np.mean(sum_mse_i)
print ('Avg:',avg_cost)



# w,b=sess.run([W, b])
# w,b,m,m2,l=sess.run([W, b, mse_r, mse_i, loss_d])
# print('w:\n',w)
# print('b:\n',b)
# print('m:\n',m)
# print('m2:\n',m2)
# print('l:\n',l)