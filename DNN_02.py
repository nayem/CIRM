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

tf.logging.set_verbosity(tf.logging.INFO)

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

        # Parameters
        self.LEARNING_RATE = 0.00001
        self.Momentum_R = 0.5  # momentum = 0.5 (for 1-5 epochs), 0.9 (for rest)
        self.TOTAL_EPOCHS = 80
        self.BATCH_SIZE = 512
        self.DISPLAY_STEP = 1

        # Network Parameters
        self.n_input = 1230  # MNIST data input (img shape: 28*28)
        self.n_hidden_1 = 1024  # 1st layer number of neurons
        self.n_hidden_2 = 1024  # 2nd layer number of neurons
        self.n_hidden_3 = 1024  # 3rd layer number of neurons
        self.n_classes = (963 + 963)  # total classes (real+imaginary)

        self.validation_error = np.full((1), np.inf)
        self.min_validation_error = np.full((1), np.inf)
        self.PREVIOUS_10 = 10
        self.DIFF_THRESHOLD = 0.000001


        # Store layers weight & bias
        self.best_weights = dict()
        self.best_biases = dict()

        with h5py.File(FILE, 'r') as f:
            key_list = list(f.keys())
            print(key_list)

            for k, v in f.items():

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
                        print(n_s[0])

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
            print(list(f.keys()))
            for k, v in f.items():
                if k == 'trData':
                    self.trData = np.transpose(np.array(v))
                    # print('trData: ', self.trData[0, 0], self.trData[0, 1], self.trData[1, 4])
                    # self.opts_dict[k] = self.trData
                elif k == 'trLabel_i':
                    self.trLabel_i = np.transpose(np.array(v))
                    # print('trLabel_i: ', self.trLabel_i[0, 0], self.trLabel_i[0, 1], self.trLabel_i[1, 4])
                    # self.opts_dict[k] = self.trLabel_i
                elif k == 'trLabel_r':
                    self.trLabel_r = np.transpose(np.array(v))
                    # print('trLabel_r: ', self.trLabel_r[0, 0], self.trLabel_r[0, 1], self.trLabel_r[1, 6])
                    # self.opts_dict[k] = self.trLabel_r
                elif k == 'cvData':
                    self.cvData = np.transpose(np.array(v))
                    # print('cvData: ', self.cvData[0, 0], self.cvData[0, 1], self.cvData[1, 5])
                    # self.opts_dict[k] = self.cvData
                elif k == 'cvLabel_i':
                    self.cvLabel_i = np.transpose(np.array(v))
                    # print('cvLabel_i: ', self.cvLabel_i[0, 0], self.cvLabel_i[0, 1], self.cvLabel_i[1, 8])
                    # self.opts_dict[k] = self.cvLabel_i
                elif k == 'cvLabel_r':
                    self.cvLabel_r = np.transpose(np.array(v))
                    # print('cvLabel_r: ', self.cvLabel_r[0, 0], self.cvLabel_r[0, 1], self.cvLabel_r[1, 6])

    def next_batch(self, batch_size, isTrainCycle=True):
        if isTrainCycle:
            selected_indics = np.random.randint(195192, size=batch_size)
        else:
            selected_indics = np.random.randint(44961, size=batch_size)

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
                    y = np.array([np.concatenate((opts.trLabel_r[indx], opts.trLabel_i[indx]), axis=0)])
                    isFirst = False
                else:
                    x = np.array([opts.cvData[indx]])
                    y = np.array([np.concatenate((opts.cvLabel_r[indx], opts.cvLabel_i[indx]), axis=0)])
                    isFirst = False

            else:
                if isTrainCycle:
                    np.append(x, [opts.trData[indx]], axis=0)
                    np.append(y, [np.concatenate((opts.trLabel_r[indx], opts.trLabel_i[indx]), axis=0)], axis=0)
                    isFirst = False
                else:
                    np.append(x, [opts.cvData[indx]], axis=0)
                    np.append(y, [np.concatenate((opts.cvLabel_r[indx], opts.cvLabel_i[indx]), axis=0)], axis=0)
                    isFirst = False

        return [x, y]


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


################################################################

'''
DNN 3 layer
'''

def cIRM_model_fn(opts, features, labels, mode):
    """Model function for cIRM."""

    # Dense Layer
    dense1 = tf.layers.dense(inputs=features, units=opts.n_hidden_1, activation=tf.nn.relu, name='layer_1')
    dense2 = tf.layers.dense(inputs=dense1, units=opts.n_hidden_2, activation=tf.nn.relu, name='layer_2')
    dense3 = tf.layers.dense(inputs=dense2, units=opts.n_hidden_3, activation=tf.nn.relu, name='layer_3')

    # Output Layer
    predictions = tf.layers.dense(inputs=dense3, units=opts.n_classes, name='output_layer')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=predictions)  # Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE, initial_accumulator_value=Momentum_R)
        optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE, initial_accumulator_value=Momentum_R)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)


def main(unused_argv):
    opts = Opts(DNN_MODEL_FILE, DNN_DATA_FILE)

    print(opts.trData.shape, "opts.trData:\n",opts.trData)
    print(opts.trLabel_r.shape, "opts.trLabel_r:\n", opts.trLabel_r)
    print(opts.trLabel_i.shape, "opts.trLabel_i:\n", opts.trLabel_i)

    print(opts.cvData.shape, "opts.cvData:\n", opts.cvData)
    print(opts.cvLabel_r.shape, "opts.cvData:\n", opts.cvLabel_r)
    print(opts.cvLabel_i.shape, "opts.cvData:\n", opts.cvLabel_i)

    # Create the Estimator
    cIRM_classifier = tf.estimator.Estimator(
        model_fn=cIRM_model_fn, model_dir="/tmp/cIRM_dense_model_02")

    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=1)

    # Train the model
    batch_x, batch_y = opts.next_batch(195192)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":batch_x},
        y=batch_y,
        batch_size=512,
        num_epochs=5,
        shuffle=True)

    cIRM_classifier.train(
        input_fn=train_input_fn,
        steps=2,
        hooks=[logging_hook])

    # Evaluate the model and print results
    val_batch_x, val_batch_y = opts.next_batch(44961, False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": val_batch_x},
        y=val_batch_y,
        num_epochs=1,
        shuffle=True)
    eval_results = cIRM_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
