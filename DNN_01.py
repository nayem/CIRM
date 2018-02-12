from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math

import tensorflow as tf
import scipy.io as sio
import numpy as np
import h5py

'''
Read parameters from .mat files
save in Opt object
'''
################################################################
DNN_DATA_FILE = "./dnn_models/DNN_datas.mat"
DNN_MODEL_FILE = "./dnn_models/DNN_params.mat"
DNN_NET_FILE = "./dnn_models/DNN_net.mat"


class Opts:

    opts_dict = dict()

    def __init__(self, FILE, FILE_DATA):
        with h5py.File(FILE, 'r') as f:
            key_list = list(f.keys())
            print (key_list)

            for k,v in f.items():

                if k == 'ARMA_order':
                    self.ARMA_order = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.ARMA_order
                elif k == 'ada_grad_eps':
                    self.ada_grad_eps = np.array(v)[0][0]
                    self.opts_dict[k] = self.ada_grad_eps
                elif k == 'ada_sgd_scale':
                    self.ada_sgd_scale = np.array(v)[0][0]
                    self.opts_dict[k] = self.ada_sgd_scale
                elif k == 'change_momentum_point':
                    self.change_momentum_point = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.change_momentum_point
                elif k == 'cost_function':
                    self.cost_function = ""
                    for c in np.array(v):
                        self.cost_function += chr(c[0])

                    self.opts_dict[k] = self.cost_function

                elif k == 'cv_interval':
                    self.cv_interval = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.cv_interval
                elif k == 'dim_input':
                    self.dim_input = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.dim_input
                elif k == 'dim_output':
                    self.dim_output = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.dim_output
                elif k == 'drop_ratio':
                    self.drop_ratio = np.array(v)[0][0]
                    self.opts_dict[k] = self.drop_ratio
                elif k == 'eval_on_gpu':
                    self.eval_on_gpu = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.eval_on_gpu
                elif k == 'final_momentum':
                    self.final_momentum = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.final_momentum
                elif k == 'hid_struct':
                    self.hid_struct = np.array(v)
                    self.opts_dict[k] = self.hid_struct
                elif k == 'initial_momentum':
                    self.initial_momentum = np.array(v)[0][0]
                    self.opts_dict[k] = self.initial_momentum
                elif k == 'isDropout':
                    self.isDropout = 0
                    self.opts_dict[k] = self.isDropout
                elif k == 'isDropoutInput':
                    self.isDropoutInput = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.isDropoutInput
                elif k == 'isGPU':
                    self.isGPU = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.isGPU
                elif k == 'isNormalize':
                    self.isNormalize = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.isNormalize
                elif k == 'isPretrain':
                    self.isPretrain = int(np.array(v)[0][0])
                    self.opts_dict[k] = self.isPretrain
                elif k == 'learner':
                    self.learner = ""
                    for c in np.array(v):
                        self.learner += chr(c[0])

                    self.opts_dict[k] = self.learner

                elif k == 'net_struct':
                    self.net_struct = np.array(v)
                    for n_s in np.array(v):
                        print (n_s[0])

                    self.opts_dict[k] = self.net_struct
                elif k == 'rbm_batch_size':
                    self.rbm_batch_size = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.rbm_batch_size
                elif k == 'rbm_learn_rate_binary':
                    self.rbm_learn_rate_binary = np.array(v)
                    # self.opts_dict[k] = self.rbm_learn_rate_binary
                elif k == 'rbm_learn_rate_real':
                    self.rbm_learn_rate_real = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.rbm_learn_rate_real
                elif k == 'rbm_max_epoch':
                    self.rbm_max_epoch = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.rbm_max_epoch
                elif k == 'save_on_fly':
                    self.save_on_fly = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.save_on_fly
                elif k == 'sgd_batch_size':
                    self.sgd_batch_size = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.sgd_batch_size
                elif k == 'sgd_learn_rate':
                    self.sgd_learn_rate = np.array(v)
                    # self.opts_dict[k] = self.sgd_learn_rate
                elif k == 'sgd_max_epoch':
                    self.sgd_max_epoch = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.sgd_max_epoch
                elif k == 'split_tanh1_c1':
                    self.split_tanh1_c1 = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.split_tanh1_c1
                elif k == 'split_tanh1_c2':
                    self.split_tanh1_c2 = int(np.array(v)[0][0])
                    # self.opts_dict[k] = self.split_tanh1_c2
                elif k == 'unit_type_hidden':
                    self.unit_type_hidden = ""
                    for c in np.array(v):
                        self.unit_type_hidden += chr(c[0])

                    # self.opts_dict[k] = self.unit_type_hidden

                elif k == 'unit_type_output':
                    self.unit_type_output = ""
                    for c in np.array(v):
                        self.unit_type_output += chr(c[0])

                    # self.opts_dict[k] = self.unit_type_output

                # elif k == 'trData':
                #     self.trData = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.trData
                # elif k == 'trLabel_i':
                #     self.trLabel_i = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.trLabel_i
                # elif k == 'trLabel_r':
                #     self.trLabel_r = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.trLabel_r
                # elif k == 'cvData':
                #     self.cvData = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.cvData
                # elif k == 'cvLabel_i':
                #     self.cvLabel_i = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.cvLabel_i
                # elif k == 'cvLabel_r':
                #     self.cvLabel_r = np.transpose( np.array(v) )
                #     # self.opts_dict[k] = self.cvLabel_r

        with h5py.File(FILE_DATA, 'r') as f:
            print ( list(f.keys()) )
            for k, v in f.items():
                if k == 'trData':
                    self.trData = np.transpose(np.array(v))
                    print ('trData: ',self.trData[0,0], self.trData[0,1], self.trData[1,4])
                    # self.opts_dict[k] = self.trData
                elif k == 'trLabel_i':
                    self.trLabel_i = np.transpose(np.array(v))
                    print('trLabel_i: ',self.trLabel_i[0, 0], self.trLabel_i[0, 1], self.trLabel_i[1,4])
                    # self.opts_dict[k] = self.trLabel_i
                elif k == 'trLabel_r':
                    self.trLabel_r = np.transpose(np.array(v))
                    print('trLabel_r: ',self.trLabel_r[0, 0], self.trLabel_r[0, 1], self.trLabel_r[1,6])
                    # self.opts_dict[k] = self.trLabel_r
                elif k == 'cvData':
                    self.cvData = np.transpose(np.array(v))
                    print('cvData: ',self.cvData[0, 0], self.cvData[0, 1], self.cvData[1,5])
                    # self.opts_dict[k] = self.cvData
                elif k == 'cvLabel_i':
                    self.cvLabel_i = np.transpose(np.array(v))
                    print('cvLabel_i: ',self.cvLabel_i[0, 0], self.cvLabel_i[0, 1], self.cvLabel_i[1,8])
                    # self.opts_dict[k] = self.cvLabel_i
                elif k == 'cvLabel_r':
                    self.cvLabel_r = np.transpose(np.array(v))
                    print('cvLabel_r: ',self.cvLabel_r[0, 0], self.cvLabel_r[0, 1], self.cvLabel_r[1,6])


    def next_batch(self, batch_size, isTrainCycle = True):
        if isTrainCycle:
            selected_indics = np.random.randint(195192, size= batch_size)
        else:
            selected_indics = np.random.randint(44961, size= batch_size)

        isFirst = True
        # print (np.shape(opts.trData) )
        # print(np.shape(opts.trLabel_r))
        # print(np.shape(opts.trLabel_i))
        # print(np.shape(opts.cvData))
        # print(np.shape(opts.cvLabel_r))
        # print(np.shape(opts.cvLabel_i))

        # print (selected_indics)

        for indx in selected_indics:
            if isFirst:
                if isTrainCycle:
                    x = np.array([opts.trData[indx]])
                    y = np.array([ np.concatenate( (opts.trLabel_r[indx], opts.trLabel_i[indx]), axis=0 ) ])
                    isFirst = False
                else:
                    x = np.array([opts.cvData[indx]])
                    y = np.array([np.concatenate((opts.cvLabel_r[indx], opts.cvLabel_i[indx]), axis=0)])
                    isFirst = False

            else:
                if isTrainCycle:
                    np.append(x, [opts.trData[indx]], axis=0)
                    np.append(y, [ np.concatenate( (opts.trLabel_r[indx], opts.trLabel_i[indx]), axis=0 ) ], axis=0)
                    isFirst = False
                else:
                    np.append(x, [opts.cvData[indx]], axis=0)
                    np.append(y, [np.concatenate((opts.cvLabel_r[indx], opts.cvLabel_i[indx]), axis=0)], axis=0)
                    isFirst = False

        return [x, y]

opts = Opts(DNN_MODEL_FILE, DNN_DATA_FILE)
# print (opts.net_struct, opts.change_momentum_point)

[x,y] = opts.next_batch(5)

# print ('x:',x)
# print ('y:',y)
# print ('x[0]:',x[0])
# print ('y[0]:',y[0])

################################################################

'''
DNN 3 layer
'''


def weight_variable (shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable (shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Create model
def multilayer_NN(x):

    layer_1 = tf.matmul(x, weights['h1']) + biases['b1']

    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']

    layer_3 = tf.matmul(layer_2, weights['h3']) + biases['b3']

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


def calc(x, y):
    # Returns predictions and error
    predictions = multilayer_NN(X)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.square(y - predictions))
    # loss = tf.reduce_mean(tf.abs(y - predictions))

    return [predictions, loss]

def write_file(best_weights, best_biases, DNN_NET_FILE):
    W_1, W_2, W_3 = [best_weights['h1'], best_weights['h2'], best_weights['h3']]
    # W_1,W_2, W_3 = sess.run([best_weights['h1'], best_weights['h2'], best_weights['h3']] )
    # W_1, W_2, W_3 = sess.run([weights['h1'], weights['h2'], weights['h3']])
    W_1, W_2, W_3, W_4 = [np.transpose(W_1), np.transpose(W_2), np.transpose(W_3), np.array([])]
    print('W_1: ', W_1.shape, 'W_2: ', W_2.shape, 'W_3: ', W_3.shape, 'W_4: ', W_4.shape)

    b_1, b_2, b_3 = [best_biases['b1'], best_biases['b2'], best_biases['b3']]
    # b_1, b_2, b_3 = sess.run([best_biases['b1'], best_biases['b2'], best_biases['b3']])
    # b_1, b_2, b_3 = sess.run([biases['b1'], biases['b2'], biases['b3']])
    b_1, b_2, b_3, b_4 = [np.reshape(b_1, (1024, 1)), np.reshape(b_2, (1024, 1)), np.reshape(b_3, (1024, 1)),
                          np.array([])]
    print('b_1: ', b_1.shape, 'b_2: ', b_2.shape, 'b_3: ', b_3.shape, 'b_4: ', b_4.shape)

    Wo, bo = [best_weights['out'], best_biases['out']]
    # Wo, bo = sess.run([best_weights['out'], best_biases['out']])
    # Wo, bo = sess.run([weights['out'], biases['out']])
    Wo, bo = [np.transpose(Wo), np.transpose(bo)]

    Wo1_1, Wo1_2, Wo1_3, Wo1_4 = [np.array([]), np.array([]), np.array([]), Wo[:963]]
    bo1_1, bo1_2, bo1_3, bo1_4 = [np.array([]), np.array([]), np.array([]),
                                  np.reshape(np.transpose(bo[:963]), (963, 1))]

    Wo2_1, Wo2_2, Wo2_3, Wo2_4 = [np.array([]), np.array([]), np.array([]), Wo[963:]]
    bo2_1, bo2_2, bo2_3, bo2_4 = [np.array([]), np.array([]), np.array([]),
                                  np.reshape(np.transpose(bo[963:]), (963, 1))]

    print('Wo1_1: ', Wo1_1.shape, 'Wo1_2: ', Wo1_2.shape, 'Wo1_3: ', Wo1_3.shape, 'Wo1_4: ', Wo1_4.shape)
    print('bo1_1: ', bo1_1.shape, 'bo1_2: ', bo1_2.shape, 'bo1_3: ', bo1_3.shape, 'bo1_4: ', bo1_4.shape)

    print('Wo2_1: ', Wo2_1.shape, 'Wo2_2: ', Wo2_2.shape, 'Wo2_3: ', Wo2_3.shape, 'Wo2_4: ', Wo2_4.shape)
    print('bo2_1: ', bo2_1.shape, 'bo2_2: ', bo2_2.shape, 'bo2_3: ', bo2_3.shape, 'bo2_4: ', bo2_4.shape)

    # Param_Dict = {'W': np.transpose([W_1,W_2, W_3]), 'b':np.transpose([b_1,b_2, b_3]) }
    # Param_Dict = {'W': { (W_1, W_2, (W_3)}, 'b': {(b_1), (b_2), (b_3)}}
    Param_Dict = np.core.records.fromarrays(
        [[W_1, W_2, W_3, W_4], [b_1, b_2, b_3, b_4], [Wo1_1, Wo1_2, Wo1_3, Wo1_4], [bo1_1, bo1_2, bo1_3, bo1_4],
         [Wo2_1, Wo2_2, Wo2_3, Wo2_4], [bo2_1, bo2_2, bo2_3, bo2_4]], names='W,b,Wo1,bo1,Wo2,bo2')

    master_dict = {'struct_net': [Param_Dict]}
    sio.savemat(DNN_NET_FILE, master_dict, format='5', long_field_names=True)
#################
# x = tf.placeholder( tf.float32, shape=[None, 1230])
# y = tf.placeholder( tf.float32, shape=[None, 963+963])

# Hidden Layer1
# W_1 = weight_variable([1230, 1024])
# b_1 = bias_variable([1024])

# # Hidden Layer2
# W_2 = weight_variable([1024, 1024])
# b_2 = bias_variable([1024])
#
# # Hidden Layer3
# W_3 = weight_variable([1024, 1024])
# b_3 = bias_variable([1024])
#
# Hidden Layer4
# W_4 = weight_variable([1024, 963+963])
# b_4 = bias_variable([963+963])


# Parameters
## HAVE to use OPS PARAMS ***********
learning_rate = 0.00001
training_epochs = 80
# training_epochs = 0
batch_size = 512
display_step = 1

# Network Parameters
## HAVE to use NET_STRUCTURE ***********
n_input = 1230 # MNIST data input (img shape: 28*28)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layer number of neurons
n_classes = (963+963) #  total classes (real+imaginary)

validation_error = np.full((1), np.inf)
min_validation_error = np.full((1), np.inf)
PREVIOUS_10 = 10
DIFF_THRESHOLD = 0.000001

# with tf.device('/gpu:2'):
# tf Graph input
X = tf.placeholder( tf.float32, shape=[None, n_input])
Y = tf.placeholder( tf.float32, shape=[None, n_classes])

# Store layers weight & bias
weights = {
    'h1': weight_variable([n_input, n_hidden_1]),
    'h2': weight_variable([n_hidden_1, n_hidden_2]),
    'h3': weight_variable([n_hidden_2, n_hidden_3]),
    'out': weight_variable([n_hidden_3, n_classes])
}
biases = {
    'b1': bias_variable([n_hidden_1]),
    'b2': bias_variable([n_hidden_2]),
    'b3': bias_variable([n_hidden_3]),
    'out': bias_variable([n_classes])
}

best_weights = dict()
best_biases = dict()
# best_weights = {
#     'h1': weight_variable([n_input, n_hidden_1]),
#     'h2': weight_variable([n_hidden_1, n_hidden_2]),
#     'h3': weight_variable([n_hidden_2, n_hidden_3]),
#     'out': weight_variable([n_hidden_3, n_classes])
# }
# best_biases = {
#     'b1': bias_variable([n_hidden_1]),
#     'b2': bias_variable([n_hidden_2]),
#     'b3': bias_variable([n_hidden_3]),
#     'out': bias_variable([n_classes])
# }

y_p, loss_op = calc(X, Y)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
# train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    epoch = 0
    while True:

        ##### Create Train Batch, evaluate COST, Weight, Bias #####
        ###########################################################

        avg_cost = 0.0
        total_batch = math.ceil(195192.0/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = opts.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, epoch_w, epoch_b = sess.run([train_op, loss_op, weights, biases], feed_dict={X: batch_x, Y: batch_y})

            # Compute average loss
            avg_cost += (c / total_batch)
            # print('Epoch: ', epoch, 'iteration: ', i,  'cost: ', c, '(c / total_batch): ', (c / total_batch), 'total_batch: ', total_batch, 'avg_cost: ', avg_cost)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))


        ##### Validation in whole batch #####
        #####################################

        val_batch_x, val_batch_y = opts.next_batch(44961, False)
        valid_cost = calc(val_batch_x, val_batch_y)[1]

        validation_cost_epoch = sess.run(valid_cost, feed_dict={X: val_batch_x, Y: val_batch_y})
        print(epoch, 'Validation error =', validation_cost_epoch)


        ##### Min validation error, update weights, bias #####
        if np.isnan(validation_cost_epoch):
            validation_cost_epoch = np.inf

        if epoch>0:
            validation_error = np.append(validation_error,validation_cost_epoch)
            min_validation_error = np.append(min_validation_error, min(validation_error))
        else:
            validation_error[epoch] = validation_cost_epoch
            min_validation_error[epoch] = min(validation_error)

        # print('Validation error =', sess.run(valid_cost, feed_dict={X: val_batch_x, Y: val_batch_y}))
        print('Best Validation error =',  min_validation_error[-1])
        if epoch >= 1 and min_validation_error[-2] > min_validation_error[-1]:
            best_weights=epoch_w.copy()
            best_biases=epoch_b.copy()

            print("Inside: ", end='')
            print(epoch_w.keys())
            write_file(epoch_w, epoch_b, DNN_NET_FILE)
            print(" File Write complete")

        # Stopping Training for stable min error
        if epoch >= training_epochs:
            # print("Last 10: ", end='')
            # print(min_validation_error[-PREVIOUS_10:])
            # print(validation_error[-PREVIOUS_10:])
            # # print(np.ediff1d(validation_error[-10:]))
            # print("Abs Diff: ", end='')
            # print(np.absolute(np.ediff1d(min_validation_error[-PREVIOUS_10:])))
            # print("Sum Abs Diff: ", end='')
            # print( sum(np.absolute(np.ediff1d(min_validation_error[-PREVIOUS_10:]))) )
            # print("Decision: ", end='')
            # print(sum(np.absolute(np.ediff1d(min_validation_error[-PREVIOUS_10:]))) < DIFF_THRESHOLD)

            if sum(np.absolute(np.ediff1d(min_validation_error[-PREVIOUS_10:]))) < DIFF_THRESHOLD:
                break


        epoch += 1

    print("Optimization Finished!")

    # val_batch_x, val_batch_y = opts.next_batch(44961, False)
    # valid_cost = calc( val_batch_x, val_batch_y )[1]
    #
    # print('Validation error =', sess.run(valid_cost, feed_dict={X: val_batch_x, Y: val_batch_y}) )

    #############################
    print("Outside: ", end='')
    print(best_weights.keys())
    W_1, W_2, W_3 = [best_weights['h1'], best_weights['h2'], best_weights['h3']]
    # W_1,W_2, W_3 = sess.run([best_weights['h1'], best_weights['h2'], best_weights['h3']] )
    # W_1, W_2, W_3 = sess.run([weights['h1'], weights['h2'], weights['h3']])
    W_1, W_2, W_3, W_4 = [ np.transpose(W_1), np.transpose(W_2), np.transpose(W_3), np.array([]) ]
    print('W_1: ', W_1.shape, 'W_2: ',W_2.shape, 'W_3: ',W_3.shape, 'W_4: ',W_4.shape)

    b_1, b_2, b_3 = [best_biases['b1'], best_biases['b2'], best_biases['b3']]
    # b_1, b_2, b_3 = sess.run([best_biases['b1'], best_biases['b2'], best_biases['b3']])
    # b_1, b_2, b_3 = sess.run([biases['b1'], biases['b2'], biases['b3']])
    b_1, b_2, b_3, b_4 = [ np.reshape(b_1,(1024,1) ) , np.reshape(b_2,(1024,1)), np.reshape(b_3,(1024,1)), np.array([]) ]
    print('b_1: ',b_1.shape,'b_2: ', b_2.shape, 'b_3: ', b_3.shape, 'b_4: ', b_4.shape)

    Wo, bo = [best_weights['out'], best_biases['out']]
    # Wo, bo = sess.run([best_weights['out'], best_biases['out']])
    # Wo, bo = sess.run([weights['out'], biases['out']])
    Wo, bo = [ np.transpose(Wo), np.transpose(bo) ]

    Wo1_1, Wo1_2, Wo1_3, Wo1_4 = [ np.array([]), np.array([]), np.array([]), Wo[:963] ]
    bo1_1, bo1_2, bo1_3, bo1_4 = [ np.array([]), np.array([]), np.array([]), np.reshape(np.transpose(bo[:963]), (963,1)) ]

    Wo2_1, Wo2_2, Wo2_3, Wo2_4 = [ np.array([]), np.array([]), np.array([]), Wo[963:] ]
    bo2_1, bo2_2, bo2_3, bo2_4 = [ np.array([]), np.array([]), np.array([]), np.reshape(np.transpose(bo[963:]),(963,1)) ]

    print('Wo1_1: ', Wo1_1.shape, 'Wo1_2: ', Wo1_2.shape, 'Wo1_3: ', Wo1_3.shape, 'Wo1_4: ', Wo1_4.shape)
    print('bo1_1: ', bo1_1.shape, 'bo1_2: ', bo1_2.shape, 'bo1_3: ', bo1_3.shape, 'bo1_4: ', bo1_4.shape)

    print('Wo2_1: ', Wo2_1.shape, 'Wo2_2: ', Wo2_2.shape, 'Wo2_3: ', Wo2_3.shape, 'Wo2_4: ', Wo2_4.shape)
    print('bo2_1: ', bo2_1.shape, 'bo2_2: ', bo2_2.shape, 'bo2_3: ', bo2_3.shape, 'bo2_4: ', bo2_4.shape)

    # Param_Dict = {'W': np.transpose([W_1,W_2, W_3]), 'b':np.transpose([b_1,b_2, b_3]) }
    # Param_Dict = {'W': { (W_1, W_2, (W_3)}, 'b': {(b_1), (b_2), (b_3)}}
    Param_Dict = np.core.records.fromarrays([ [W_1,W_2, W_3, W_4], [b_1,b_2, b_3, b_4], [Wo1_1,Wo1_2, Wo1_3, Wo1_4],[bo1_1, bo1_2, bo1_3, bo1_4], [Wo2_1, Wo2_2, Wo2_3, Wo2_4],[bo2_1, bo2_2, bo2_3, bo2_4] ], names='W,b,Wo1,bo1,Wo2,bo2')

    master_dict = {'struct_net': [Param_Dict] }
    sio.savemat(DNN_NET_FILE, master_dict, format='5', long_field_names=True)

