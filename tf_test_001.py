from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math

import tensorflow as tf
import scipy.io as sio
import numpy as np

def fun(m):
    print (m.shape)
    return m

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

X = tf.placeholder(tf.float32)
m = tf.placeholder(tf.float32)
t_fun = fun(m)

# training loop
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10):
    if i < 5:
        print( sess.run([t_fun], feed_dict={m:0.5}) )
    else:
        print( sess.run([t_fun], feed_dict={m: 0.9}) )

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))