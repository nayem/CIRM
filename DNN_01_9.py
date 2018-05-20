# DNN_01_9.py
# Matlab initialization strategy
# Train and Dev different loss functions
# Adam Optimizer
#
# Apr 01, 2018
#



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math

import scipy.io as sio
import numpy as np
import h5py
import tensorflow as tf
import sys

import time

'''
Read parameters from .mat files
save in Opt object
'''
################################################################

DNN_DATA_FILE = "./dnn_models/DNN_datas_05.mat"
# DNN_DATA_FILE = "./dnn_models/BR2_DNN_datas.mat"

DNN_MODEL_FILE = "./dnn_models/DNN_params_05.mat"
# DNN_MODEL_FILE = "./dnn_models/BR2_DNN_params.mat"

DNN_NET_FILE = "./dnn_models/DNN_net_09.mat"

# ModelFN_FILE = "./dnn_models/dnncirm.noiseSSN_05.mat"


class Opts:
    opts_dict = dict()

    def __init__(self, FILE, FILE_DATA):
        with h5py.File(FILE, 'r') as f:
            key_list = list(f.keys())
            print('Opt key:',key_list)

            for k, v in f['opts'].items():

                print('key:', k)

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
                        print('Opts Net Stuct:',n_s[0])

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

                elif k == 'unit_type_output':
                    self.unit_type_output = ""
                    for c in np.array(v):
                        self.unit_type_output += chr(c[0])

        with h5py.File(FILE_DATA, 'r') as f:
            print('Opts h5py keys:', list(f.keys()))
            for k, v in f.items():

                if k == 'trData':
                    self.trData = np.transpose(np.array(v))
                elif k == 'trLabel_i':
                    self.trLabel_i = np.transpose(np.array(v))
                elif k == 'trLabel_r':
                    self.trLabel_r = np.transpose(np.array(v))
                elif k == 'cvData':
                    self.cvData = np.transpose(np.array(v))
                elif k == 'cvLabel_i':
                    self.cvLabel_i = np.transpose(np.array(v))
                elif k == 'cvLabel_r':
                    self.cvLabel_r = np.transpose(np.array(v))

            self.trLabel = np.concatenate((self.trLabel_r, self.trLabel_i), axis=1)
            self.cvLabel = np.concatenate((self.cvLabel_r, self.cvLabel_i), axis=1)

        # with h5py.File(ModelFN_FILE, 'r') as f:
        #
        #     print('Opts h5py keys:', list(f.keys()))
        #     for k, v in f.items():
        #
        #         if k == 'net':
        #             # ref_w_1 = v['W'][0, 0]
        #             # print( f[v['W'][0,0]].value )
        #
        #             # W weights
        #             ref_w_1 = v['W'][0,0]
        #             self.pre_w_1 = np.array(f[ref_w_1].value , ndmin=2).T
        #             print('pre_w_1: ', self.pre_w_1.shape)
        #
        #             ref_w_2 = v['W'][0, 1]
        #             self.pre_w_2 = np.array(f[ref_w_2].value,  ndmin=2).T
        #             print('pre_w_2: ', self.pre_w_2.shape)
        #
        #             ref_w_3 = v['W'][0, 2]
        #             self.pre_w_3 = np.array(f[ref_w_3].value, ndmin=2).T
        #             print('pre_w_3: ', self.pre_w_3.shape)
        #
        #             ref_w_4 = v['W'][0, 3]
        #             self.pre_w_4 = np.array(f[ref_w_4].value, ndmin=2).T
        #             print('pre_w_4: ', self.pre_w_4.shape)
        #
        #             # b weights
        #             ref_b_1 = v['b'][0, 0]
        #             self.pre_b_1 = np.array(f[ref_b_1].value, ndmin=2).T
        #             print('pre_b_1: ', self.pre_b_1.shape)
        #
        #             ref_b_2 = v['b'][0, 1]
        #             self.pre_b_2 = np.array(f[ref_b_2].value, ndmin=2).T
        #             print('pre_b_2: ', self.pre_b_2.shape)
        #
        #             ref_b_3 = v['b'][0, 2]
        #             self.pre_b_3 = np.array(f[ref_b_3].value, ndmin=2).T
        #             print('pre_b_3: ', self.pre_b_3.shape)
        #
        #             ref_b_4 = v['b'][0, 3]
        #             self.pre_b_4 = np.array(f[ref_b_4].value, ndmin=2).T
        #             print('pre_b_4: ', self.pre_b_4.shape)
        #
        #             # Wo1 weights
        #             ref_wo1_4 = v['Wo1'][0, 3]
        #             self.pre_wo1_4 = np.array(f[ref_wo1_4].value, ndmin=2).T
        #             print('pre_w01_2: ', self.pre_wo1_4.shape)
        #
        #             # bo1 weights
        #             ref_bo1_4 = v['bo1'][0, 3]
        #             self.pre_bo1_4 = np.array(f[ref_bo1_4].value, ndmin=2).T
        #             print('pre_bo1_2: ', self.pre_bo1_4.shape)
        #
        #             # Wo2 weights
        #             ref_wo2_4 = v['Wo2'][0, 3]
        #             self.pre_wo2_4 = np.array(f[ref_wo2_4].value, ndmin=2).T
        #             print('pre_wo2_4: ', self.pre_wo2_4.shape)
        #
        #             # bo2 weights
        #             ref_bo2_4 = v['bo2'][0, 3]
        #             self.pre_bo2_4 = np.array(f[ref_bo2_4].value, ndmin=2).T
        #             print('pre_bo2_4: ', self.pre_bo2_4.shape)



    def ready_batchID(self, total_num_samples, batch_size):
        # TRAIN: total_num_samples = self.trData.shape[0] = 195192
        # DEV: total_num_samples = self.trData.shape[0] = 44961

        batchID = []
        num_batch = math.ceil(total_num_samples/batch_size)

        for b in range( int(num_batch) ):
            s = b*batch_size
            e = (b+1)*batch_size -1

            if e >= total_num_samples:
                e = total_num_samples - 1

            batchID.append((s,e))

        return np.array(batchID,ndmin=2)


    def suffle_data(self, total_num_samples):
        # TRAIN: total_num_samples = self.trData.shape[0] = 195192
        # DEV: total_num_samples = self.trData.shape[0] = 44961

        return  np.random.permutation(total_num_samples)


    def next_batch(self, total_num_samples, batch_size, isTrainCycle=True):
        # TRAIN: total_num_samples = self.trData.shape[0] = 195192
        # DEV: total_num_samples = self.trData.shape[0] = 44961

        batchID = self.ready_batchID(total_num_samples, batch_size)
        seq = self.suffle_data(total_num_samples)

        for batch in range(batchID.shape[0]):
            if isTrainCycle:
                x = opts.trData[ seq[batchID[batch][0]:batchID[batch][1] ] ]
                y = opts.trLabel[ seq[batchID[batch][0]:batchID[batch][1] ] ]

            else:
                x = opts.cvData[seq[batchID[batch][0]:batchID[batch][1]]]
                y = opts.cvLabel[seq[batchID[batch][0]:batchID[batch][1]]]

            # print('Next Batch', x.shape, y.shape)
            yield [x, y]


    def next_batch2(self, batch_size, isTrainCycle=True):
        if isTrainCycle:
            # selected_indics = np.random.choice(10, size=batch_size, replace=False)
            selected_indics = np.random.randint(195192, size=batch_size)
        else:
            # selected_indics = np.random.randint(7, size= batch_size)
            selected_indics = np.random.randint(44961, size=batch_size)

        # print ("selected_indics: ",selected_indics)
        if isTrainCycle:
            x = opts.trData[selected_indics]
            y = np.concatenate((opts.trLabel_r, opts.trLabel_i), axis=1)[selected_indics]

        else:
            x = opts.cvData[selected_indics]
            y = np.concatenate((opts.cvLabel_r, opts.cvLabel_i), axis=1)[selected_indics]

        # print('Next Batch', x.shape, y.shape)
        return [x, y]


opts = Opts(DNN_MODEL_FILE, DNN_DATA_FILE)

# for e, (x,y) in enumerate(opts.next_batch(opts.trData.shape[0],opts.sgd_batch_size)):
#     print(e, x[0:5],y[0:5])
# quit()

# print(opts.trData.shape, "opts.trData:\n",opts.trData)
# print(opts.trLabel_r.shape, "opts.trLabel_r:\n", opts.trLabel_r)
# print(opts.trLabel_i.shape, "opts.trLabel_i:\n", opts.trLabel_i)
#
# print(opts.cvData.shape, "opts.cvData:\n", opts.cvData)
# print(opts.cvLabel_r.shape, "opts.cvLabel_r:\n", opts.cvLabel_r)
# print(opts.cvLabel_i.shape, "opts.cvLabel_i:\n", opts.cvLabel_i)



################################################################

# Parameters
## HAVE to use OPS PARAMS ***********
learning_rate = 0.001
# training_epochs = 80
# batch_size = 256
display_step = 1

# Network Parameters
## HAVE to use NET_STRUCTURE ***********
n_input_dim = 195192.0
n_input = 1230  # data input
n_hidden_1 = 1024  # 1st layer number of neurons
n_hidden_2 = 1024  # 2nd layer number of neurons
n_hidden_3 = 1024  # 3rd layer number of neurons
n_classes = (963 + 963)  # total classes (real+imaginary)

# For checking
# train_time = np.zeros(training_epochs)

# validation_error = np.full((1), np.inf)
# min_validation_error = np.full((1), np.inf)

Best_Cost = - np.inf
Best_Weight, Best_Bias = None, None
Best_epoch = -1

# For displaying
PREVIOUS_10 = 10
DIFF_THRESHOLD = 0.000001


################################################################
'''
Fully-Connected 3 layer Feed Forward Net
'''

X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, n_classes])


# Xavier Initialization
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


# Store layers weight & bias

# weights = {
#     'h1': tf.Variable(opts.pre_w_1.T),
#     'h2': tf.Variable(opts.pre_w_2.T),
#     'h3': tf.Variable(opts.pre_w_3.T),
#     'out': tf.Variable(np.concatenate( (opts.pre_wo1_4,opts.pre_wo2_4), axis=0).T)
# }
# biases = {
#     'b1': tf.Variable(opts.pre_b_1.T),
#     'b2': tf.Variable(opts.pre_b_2.T),
#     'b3': tf.Variable(opts.pre_b_3.T),
#     'out': tf.Variable(np.concatenate( (opts.pre_bo1_4,opts.pre_bo2_4), axis=0).T)
# }

# old way
# weights = {
#     'h1': weight_variable([n_input, n_hidden_1],0.001),
#     'h2': weight_variable([n_hidden_1, n_hidden_2],0.001),
#     'h3': weight_variable([n_hidden_2, n_hidden_3],0.001),
#     'out': weight_variable([n_hidden_3, n_classes],0.001,0.0)
# }
# biases = {
#     'b1': bias_variable([1, n_hidden_1],0.0),
#     'b2': bias_variable([1, n_hidden_2],0.0),
#     'b3': bias_variable([1, n_hidden_3],0.0),
#     'out': bias_variable([1, n_classes],0.001,0.0)
# }

weights = {
    'h1': tf.get_variable('weight_h1', shape=[n_input, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable('weight_h2',shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'h3': tf.get_variable('weight_h3',shape=[n_hidden_2, n_hidden_3], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('weight_hout',shape=[n_hidden_3, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.get_variable('weight_b1',shape=[1, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable('weight_b2',shape=[1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable('weight_b3',shape=[1, n_hidden_3], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('weight_bout',shape=[1, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}


# Create model
def multilayer_NN(x):
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


def calc(x, y):
    # Returns predictions and error
    predictions = multilayer_NN(x)

    # Define loss and optimizer
    r, c = y.get_shape().as_list()
    r = opts.sgd_batch_size
    print('r:', type(r), r, ',c:', type(c), c)

    # TRAIN: matlab
    # cost1 = 0.5 * sum(sum((pred_real - label_real). ^ 2)) / num_sample;
    # cost2 = 0.5 * sum(sum((pred_imag - label_imag). ^ 2)) / num_sample;
    # cost = cost1 + cost2;
    cost1 = tf.reduce_sum(tf.squared_difference(y[:, :c//2], predictions[:, :c//2]))
    cost2 = tf.reduce_sum(tf.squared_difference(y[:, c//2:], predictions[:, c//2:]))
    loss_t = 0.5*(cost1+cost2)/r
    # mse = tf.losses.mean_squared_error(labels=y, predictions=predictions,weights=0.5)
    # loss_t = tf.divide(tf.reduce_sum(mse), r)


    # DEV: matlab
    # dev_perfs = -mean(sum((dev_label_real - dev_netout1). ^ 2)) - mean(sum((dev_label_imag - dev_netout2). ^ 2));
    mse_r = tf.reduce_sum(tf.squared_difference(y[:, :c // 2], predictions[:, :c // 2]), axis=0)
    mse_i = tf.reduce_sum(tf.squared_difference(y[:, c // 2:], predictions[:, c // 2:]), axis=0)

    loss_d = -tf.reduce_mean(mse_r)-tf.reduce_mean(mse_i)


    return [predictions, loss_t, mse_r, mse_i, loss_d ]


def write_file(best_weights, best_biases, DNN_NET_FILE):
    W_1, W_2, W_3, W_4 = [np.array(best_weights['h1'], ndmin=2), np.array(best_weights['h2'], ndmin=2),
                          np.array(best_weights['h3'], ndmin=2), np.array([], ndmin=2)]
    W_1, W_2, W_3, W_4 = W_1.T, W_2.T, W_3.T, W_4

    print('W_1: ', W_1.shape, 'W_2: ', W_2.shape, 'W_3: ', W_3.shape, 'W_4: ', W_4.shape)

    b_1, b_2, b_3, b_4 = [np.array(best_biases['b1'], ndmin=2), np.array(best_biases['b2'], ndmin=2),
                          np.array(best_biases['b3'], ndmin=2), np.array([], ndmin=2)]
    b_1, b_2, b_3, b_4 = b_1.T, b_2.T, b_3.T, b_4

    print('b_1: ', b_1.shape, 'b_2: ', b_2.shape, 'b_3: ', b_3.shape, 'b_4: ', b_4.shape)

    Wo, bo = np.array(best_weights['out'], ndmin=2), np.array(best_biases['out'], ndmin=2)

    Wo1_1, Wo1_2, Wo1_3, Wo1_4 = [np.array([], ndmin=2), np.array([], ndmin=2), np.array([], ndmin=2), Wo[:, :963].T]
    bo1_1, bo1_2, bo1_3, bo1_4 = [np.array([], ndmin=2), np.array([], ndmin=2), np.array([], ndmin=2),
                                  bo[:, :963].T]
    # Wo1_1, Wo1_2, Wo1_3, Wo1_4 = [np.array([], ndmin=2), np.array([], ndmin=2), np.array([], ndmin=2), Wo[:963]]
    # bo1_1, bo1_2, bo1_3, bo1_4 = [np.array([],ndmin=2), np.array([],ndmin=2), np.array([],ndmin=2),
    #                               np.reshape(np.transpose(bo[:963]), (963, 1))]


    Wo2_1, Wo2_2, Wo2_3, Wo2_4 = [np.array([], ndmin=2), np.array([], ndmin=2), np.array([], ndmin=2), Wo[:, 963:].T]
    bo2_1, bo2_2, bo2_3, bo2_4 = [np.array([], ndmin=2), np.array([], ndmin=2), np.array([], ndmin=2),
                                  bo[:, 963:].T]
    # Wo2_1, Wo2_2, Wo2_3, Wo2_4 = [np.array([],ndmin=2), np.array([],ndmin=2), np.array([],ndmin=2), Wo[963:]]
    # bo2_1, bo2_2, bo2_3, bo2_4 = [np.array([],ndmin=2), np.array([],ndmin=2), np.array([],ndmin=2),
    #                               np.reshape(np.transpose(bo[963:]), (963, 1))]

    print('Wo1_1: ', Wo1_1.shape, 'Wo1_2: ', Wo1_2.shape, 'Wo1_3: ', Wo1_3.shape, 'Wo1_4: ', Wo1_4.shape)
    print('bo1_1: ', bo1_1.shape, 'bo1_2: ', bo1_2.shape, 'bo1_3: ', bo1_3.shape, 'bo1_4: ', bo1_4.shape)

    print('Wo2_1: ', Wo2_1.shape, 'Wo2_2: ', Wo2_2.shape, 'Wo2_3: ', Wo2_3.shape, 'Wo2_4: ', Wo2_4.shape)
    print('bo2_1: ', bo2_1.shape, 'bo2_2: ', bo2_2.shape, 'bo2_3: ', bo2_3.shape, 'bo2_4: ', bo2_4.shape)

    # Param_Dict = {'W': np.transpose([W_1,W_2, W_3]), 'b':np.transpose([b_1,b_2, b_3]) }
    # Param_Dict = {'W': { (W_1, W_2, (W_3)}, 'b': {(b_1), (b_2), (b_3)}}
    Param_Dict = np.core.records.fromarrays(
        [[W_1, W_2, W_3, W_4], [b_1, b_2, b_3, b_4], [Wo1_1, Wo1_2, Wo1_3, Wo1_4], [bo1_1, bo1_2, bo1_3, bo1_4],
         [Wo2_1, Wo2_2, Wo2_3, Wo2_4], [bo2_1, bo2_2, bo2_3, bo2_4]], names='W,b,Wo1,bo1,Wo2,bo2')

    # print(Param_Dict.shape, Param_Dict)
    master_dict = {'struct_net': [Param_Dict]}

    sio.savemat(DNN_NET_FILE, master_dict, format='5', long_field_names=True)



################################################################
'''
main()
'''

# TRAIN ops
y_p, loss_op, m_r, m_i, _ = calc(X, Y)
train_op = tf.train.AdamOptimizer().minimize(loss=loss_op)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training cycle

for epoch in range(opts.sgd_max_epoch):

    s = time.time()
    cost_sum = 0.0

    ##### Create Train Batch, evaluate COST, Weight, Bias #####
    ###########################################################
    for batch_num, (batch_x, batch_y) in enumerate(opts.next_batch(opts.trData.shape[0], opts.sgd_batch_size)):

        _, c, epoch_w, epoch_b  = sess.run([train_op, loss_op, weights, biases ],
                                          feed_dict={X: batch_x, Y: batch_y})
        cost_sum += c

        if batch_num % 100 == 0:
            print('[T] - Epoch:', epoch, ', batch_num:', batch_num, ", Cost:", c, "Cost Sum:", cost_sum)

    print('[T] - Epoch:', epoch, ",Sum:", cost_sum)

    # total_batch = int(math.ceil(n_input_dim / batch_size))
    # # total_batch = 100
    #
    # s = time.time()
    # # Loop over all batches
    # for i in range(total_batch):
    #
    #     batch_x, batch_y = opts.next_batch(batch_size)
    #
    #     _, c, epoch_w, epoch_b = sess.run([train_op, loss_op, weights, biases],
    #                                       feed_dict={X: batch_x, Y: batch_y})
    #
    #     avg_cost += c
    #     if i % 100 == 0:
    #         print('[T] - total_batch:', total_batch, ', i:', i, ", Cost:", c, ", T_avg:", avg_cost)
    #
    # # avg_cost /= total_batch
    # train_time[epoch] = time.time() - s
    #
    # # Display logs per epoch step
    # if epoch % display_step == 0:
    #     print("[T] - Epoch:", epoch, "cost=", avg_cost, "Time:", train_time[epoch])


    ################ Validation in whole batch ################
    ###########################################################

    avg_cost, sum_mse_r, sum_mse_i = 0.0, np.zeros(n_classes//2), np.zeros(n_classes//2)
    ##### Create DEV Batch, evaluate COST, Weight, Bias #####
    ###########################################################
    for batch_num, (batch_x, batch_y) in enumerate(opts.next_batch(opts.cvData.shape[0], opts.sgd_batch_size, isTrainCycle=False)):

        mse_r, mse_i = sess.run([m_r, m_i], feed_dict={X: batch_x, Y: batch_y})
        # print(sum_mse_r.shape, mse_r.shape, mse_i.shape)
        sum_mse_r += (mse_r)
        sum_mse_i += (mse_i)

    avg_cost = - np.mean(sum_mse_r) - np.mean(sum_mse_i)
    print('[D] - Epoch:', epoch, ", Mean Real:", - np.mean(sum_mse_r), ", Mean Img:" , - np.mean(sum_mse_i), ", Avg Cost:", avg_cost)

    # val_batch_x, val_batch_y = opts.next_batch(44961, False)
    #
    # val_mse = tf.losses.mean_squared_error(labels=Y, predictions=multilayer_NN(X))
    # val_loss = tf.reduce_sum(val_mse)
    #
    # validation_cost_epoch = sess.run(val_loss, feed_dict={X: val_batch_x, Y: val_batch_y})
    # print('[V] - Epoch:', epoch, 'Validation error =', validation_cost_epoch, "Time:", time.time() - s)

    ####### Min validation error, update weights, bias #########
    ############################################################


    if avg_cost > Best_Cost:
        Best_Cost = avg_cost
        Best_Weight = epoch_w
        Best_Bias = epoch_b
        Best_epoch = epoch

        print('***** [D] - Best Model at Epoch:', epoch, ", Avg Cost:", avg_cost, '*****')

    # if np.isnan(validation_cost_epoch):
    #     validation_cost_epoch = np.inf
    #
    # if epoch > 0:
    #     validation_error = np.append(validation_error, validation_cost_epoch)
    #     min_validation_error = np.append(min_validation_error, min(validation_error[-1], min_validation_error[-1]))
    # else:
    #     validation_error[epoch] = validation_cost_epoch
    #     min_validation_error[epoch] = min(validation_error)




    ######################  Write Model File ###################
    ############################################################

    print('[] - Elapsed Time  =', time.time()-s)

    # print('[V] - Best Validation error =', min_validation_error[-1])
    #
    # if epoch >= 1 and min_validation_error[-2] > min_validation_error[-1]:
    #     write_file(epoch_w, epoch_b, DNN_NET_FILE)
    #     print(" File Write complete")
    #
    # epoch += 1
    #
    # # Stopping Training for stable min error
    # if epoch >= training_epochs:
    #     break
    #     # if sum(np.absolute(np.ediff1d(min_validation_error[-PREVIOUS_10:]))) < DIFF_THRESHOLD:
    #     #     break





# Training DONE !!!
print("Optimization Finished!")
print('***** [-] - Best Model at Epoch:', Best_epoch, ", Best Val Cost:", Best_Cost, '*****')
write_file(Best_Weight, Best_Bias, DNN_NET_FILE)
print(" File Write complete")


# print(len(validation_error), "Validation Cost:")
# print(validation_error)
# print(len(min_validation_error), "min_validation_error Cost:")
# print(min_validation_error)
