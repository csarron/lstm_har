#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import time
import urllib
import zipfile
import os
import sys


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
    return [os.path.join(path, data_file) for data_file in os.listdir(path)
            if os.path.isfile(os.path.join(path, data_file))]


class Config(object):
    """
    define a class to store parameters,
    the input should be feature matrix of training and testing
    """

    def __init__(self, input_data_train, input_data_test):
        # Input data
        self.train_count = len(input_data_train)  # 7352 training series
        self.test_data_count = len(input_data_test)  # 2947 testing series
        self.n_steps = len(input_data_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 1000
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(input_data_train[0][0])  # Features count is of 9: three 3D sensors features over time
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
    return tf.nn.xw_plus_b(lstm_last_output, conf.W['output'], conf.biases['output'], name="output")


def dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%2d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("")

    print("Downloading UCI HAR Dataset...")
    data_file = "UCI HAR Dataset.zip"
    if not os.path.exists(data_file):
        urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip",
                           data_file, reporthook=dl_progress)
        print(" downloaded.")
    else:
        print("UCI HAR Dataset.zip already downloaded.")

    print("Extracting UCI HAR Dataset.zip...")
    extract_directory = os.path.abspath("data")
    if not os.path.exists(extract_directory):
        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall(extract_directory)
    else:
        print("Dataset already extracted.")


if __name__ == "__main__":
    maybe_download_data()
    print("Begin training model...")
    init_time = time.time()
    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    DATA_PATH = "data/UCI HAR Dataset/"
    SIGNALS = "Inertial Signals/"
    TRAIN = "train/"
    TEST = "test/"
    LABEL_TRAIN_FILE = "y_train.txt"
    LABEL_TEST_FILE = "y_test.txt"

    input_data_path_train = DATA_PATH + TRAIN + SIGNALS
    input_data_path_test = DATA_PATH + TEST + SIGNALS

    input_data_train_files = list_files(input_data_path_train)
    input_data_test_files = list_files(input_data_path_test)

    if not __debug__:
        print(input_data_train_files)
        print(input_data_test_files)

    input_train = read_data(input_data_train_files)
    input_test = read_data(input_data_test_files)

    label_train_file = DATA_PATH + TRAIN + LABEL_TRAIN_FILE
    label_test_file = DATA_PATH + TEST + LABEL_TEST_FILE
    y_train = read_label(label_train_file)
    y_test = read_label(label_test_file)

    # -----------------------------------
    # step2: define parameters for model
    # -----------------------------------
    config = Config(input_train, input_test)

    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="input")
    Y = tf.placeholder(tf.float32, [None, config.n_classes], name="label")

    predicted_label = lstm_net(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # softmax loss and L2
    cost = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_label, labels=Y)),
                  l2, name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    correct_predicted = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predicted, dtype=tf.float32), name="accuracy")

    # --------------------------------------------
    # step4: Train the neural network
    # --------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run()
    model_file_name = "data/lstm_har.model"
    saver = tf.train.Saver()
    best_accuracy = 0.0
    best_iter = 0
    # Start training for each batch and loop epochs
    for epoch in range(config.training_epochs):
        begin_time = time.time()
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: input_train[start:end], Y: y_train[start:end]})
        train_time = time.time()
        # Test completely at every epoch: calculate accuracy
        predict_out, accuracy_out, loss_out = sess.run([predicted_label, accuracy, cost],
                                                       feed_dict={X: input_test, Y: y_test})
        end_time = time.time()
        if accuracy_out > best_accuracy:
            best_accuracy = accuracy_out
            best_iter = epoch
            save_start_time = time.time()
            save_path = saver.save(sess, model_file_name)
            print("Model saved at: {}, takes {:6.4f}s".format(save_path, (time.time() - save_start_time)))
        print("Iter:{:3d},".format(epoch)
              + "test_acc: {:6.4f}%,".format(accuracy_out * 100)
              + " loss:{:5.3f},".format(loss_out)
              + " t_train:{:6.4f}s,".format(train_time - begin_time)
              + " t_test:{:6.4f}s,".format(end_time - train_time)
              + " best_test_acc:{:6.4f}".format(best_accuracy * 100) + " (at iter{:3d})".format(best_iter))
    print("best epoch's test accuracy: {:6.4f}".format(best_accuracy * 100) + " at iter:{:3d}".format(best_iter))
    print("finished, takes {:6.4f}s in total".format(time.time() - init_time))
