#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import time
from os import listdir
from os.path import isfile, join
import glob


def read_data(signal_files):
    """
    load sensor data into a tensor 

    :param signal_files: a list of sensor data files
    :return: a tensor represents sensor data
    """
    signal_tensor = np.dstack([np.loadtxt(signal_file) for signal_file in signal_files])
    if not __debug__:
        print('loaded tensor shape: ', signal_tensor.shape)
    return signal_tensor


def read_label(label_file):
    """
    load label data into a one-hot tensor

    :param label_file: the file that contains the label
    :return: a one-hot represented tensor
    """
    y_ = np.loadtxt(label_file, dtype=np.int32)

    if not __debug__:
        print('loaded label shape: ', y_.shape)

    # use one-hot encoding
    y_one_hot = np.zeros((y_.size, y_.max()))
    y_one_hot[np.arange(y_.size), y_ - 1] = 1

    return y_one_hot


def list_files(path):
    """
    get the files under a folder
    :param path: data path
    :return: a list of data files
    """
    return [join(path, data_file) for data_file in listdir(path) if isfile(join(path, data_file))]


class Config(object):
    """
    define a class to store parameters,
    the input should be feature matrix of training and testing
    """

    def __init__(self, input_data_test):
        # Input data
        self.test_data_count = len(input_data_test)  # 2947 testing series
        self.n_steps = len(input_data_test[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 200
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(input_data_test[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def lstm_net(feature_matrix, conf):
    """
    model a LSTM Network, it stacks 2 LSTM layers, each layer has n_hidden=32 cells
    and 1 output layer, it is a full connected layer

    :param feature_matrix: feature matrix, shape=[batch_size, time_steps, n_inputs]
    :param conf: config of network
    :return: output matrix, shape=[batch_size, n_classes]
    """

    # Exchange dim 1 and dim 0
    feature_matrix = tf.transpose(feature_matrix, [1, 0, 2])
    # New feature_mat's shape: [time_steps, batch_size, n_inputs]

    # Temporarily crush the feature_mat's dimensions
    feature_matrix = tf.reshape(feature_matrix, [-1, conf.n_inputs])
    # New feature_mat's shape: [time_steps*batch_size, n_inputs]

    # Linear activation, reshaping inputs to the LSTM's number of hidden:
    hidden = tf.nn.relu(tf.matmul(feature_matrix, conf.W['hidden']) + conf.biases['hidden'])
    # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    hidden = tf.split(hidden, conf.n_steps, 0)
    # New hidden's shape: a list of length "time_step" containing tensors of shape [batch_size, n_hidden]

    # Define LSTM cell of first hidden layer:

    # Stack two LSTM layers, both layers has the same shape
    lstm_layers = rnn.MultiRNNCell([rnn.BasicLSTMCell(conf.n_hidden, forget_bias=1.0) for _ in range(2)])

    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    outputs, _ = rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
    # outputs' shape: a list of length "time_step" containing tensors of shape [batch_size, n_classes]

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, conf.W['output']) + conf.biases['output']


if __name__ == "__main__":
    init_time = time.time()
    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    DATA_PATH = "data/UCI HAR Dataset/"
    SIGNALS = "Inertial Signals/"
    TEST = "test/"
    LABEL_TEST_FILE = "y_test.txt"

    input_data_path_test = DATA_PATH + TEST + SIGNALS
    input_data_test_files = list_files(input_data_path_test)

    if not __debug__:
        print(input_data_test_files)

    input_test = read_data(input_data_test_files)

    label_test_file = DATA_PATH + TEST + LABEL_TEST_FILE
    y_test = read_label(label_test_file)

    # -----------------------------------
    # step2: define parameters for model
    # -----------------------------------
    config = Config(input_test)

    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    predicted_label = lstm_net(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # softmax loss and L2
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_label, labels=Y)) + l2
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    correct_predicted = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predicted, dtype=tf.float32))
    session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run()
    # --------------------------------------------
    # step4: Test the neural network
    # --------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    model_file_name = "data/best_test_acc_tf1.1.model"

    init_end_time = time.time()
    print("init finished, compile graph and init variables takes {}s".format(init_end_time - init_time))

    if glob.glob(model_file_name + "*"):
        print("begin restoring")
        begin_time = time.time()
        saver = tf.train.import_meta_graph(model_file_name + ".meta")
        saver.restore(session, model_file_name)
        end_time = time.time()
        print("restored model, takes {} s".format(end_time - begin_time))
        begin_time = time.time()
        predict_out = session.run(predicted_label, feed_dict={X: input_test, Y: y_test})
        end_time = time.time()

        accuracy_out, loss_out = session.run([accuracy, cost], feed_dict={X: input_test, Y: y_test})
        print("accuracy: {}".format(accuracy_out)
              + " takes {} s".format(end_time - begin_time)
              + " loss_out: {}".format(loss_out))
    else:
        print("model not exist!")
        exit(1)

    print("Finished, takes {}s".format(time.time() - init_time))
