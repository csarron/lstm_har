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

    def __init__(self, input_data):
        # Input data
        self.train_count = len(input_data)  # 7352 training series
        self.time_steps = len(input_data[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss = 0.0015
        self.training_epochs = 1000
        self.batch_size = 1500

        # LSTM structure
        self.input_dim = len(input_data[0][0])  # Features count is of 9: three 3D sensors features over time
        self.layer_size = 2
        self.hidden_units = 32  # nb of neurons inside the neural network
        self.num_classes = 6  # Final output classes


def lstm_net(feature_matrix, conf):
    """
    model a LSTM Network, it stacks layer_size LSTM layers, each layer has hidden_units=32 cells
    and 1 output layer, it is a full connected layer
       
    :param feature_matrix: feature matrix, shape=[batch_size, time_steps, input_dim]
    :param conf: config of network
    :return: output matrix, shape=[batch_size, num_classes]
    """

    # Exchange dim 1 and dim 0
    feature_matrix = tf.transpose(feature_matrix, [1, 0, 2])
    # New feature_mat's shape: [time_steps, batch_size, input_dim]

    # Temporarily crush the feature_mat's dimensions
    feature_matrix = tf.reshape(feature_matrix, [-1, conf.input_dim], name="reshape")
    # New feature_mat's shape: [time_steps*batch_size, input_dim]

    # Linear activation, reshaping inputs to the LSTM's number of inputs:
    w_in = tf.Variable(tf.random_normal([conf.input_dim, conf.hidden_units]), name="w_in")
    b_in = tf.Variable(tf.random_normal([conf.hidden_units], mean=1.0), name="b_in")

    inputs = tf.nn.relu(tf.matmul(feature_matrix, w_in) + b_in, name="relu")

    # Split the series because the rnn cell needs time_steps features, each of shape:
    # New inputs's shape: a list of length "time_step" containing tensors of shape [batch_size, hidden_units]
    inputs = tf.split(inputs, conf.time_steps, 0)

    # Stack two LSTM layers, both layers has the same shape
    lstm_layers = rnn.MultiRNNCell([rnn.BasicLSTMCell(conf.hidden_units) for _ in range(conf.layer_size)])

    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    outputs, _ = rnn.static_rnn(lstm_layers, inputs, dtype=tf.float32)
    # outputs' shape: a list of length "time_step" containing tensors of shape [batch_size, num_classes]

    # Get last time step's output feature for a "many to one" style classifier,
    lstm_last_output = outputs[-1]

    # Linear activation
    w_out = tf.Variable(tf.random_normal([conf.hidden_units, conf.num_classes]), name="w_out")
    b_out = tf.Variable(tf.random_normal([conf.num_classes]), name="b_out")

    return tf.nn.xw_plus_b(lstm_last_output, w_out, b_out, name="output")


def dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%2d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("")

    print("Downloading UCI HAR Dataset...")
    data_file = "data/UCI HAR Dataset.zip"
    if not os.path.exists(data_file):
        if not os.path.exists("data"):
            os.mkdir("data")
        urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip",
                           data_file, reporthook=dl_progress)
        print(" downloaded.")
    else:
        print("UCI HAR Dataset.zip already downloaded.")

    print("Extracting UCI HAR Dataset.zip...")
    extract_directory = os.path.abspath("data/UCI HAR Dataset")
    if not os.path.exists(extract_directory):
        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall("data")
    else:
        print("Dataset already extracted.")


def get_data():
    data_path = "data/UCI HAR Dataset/"
    signal_path = "Inertial Signals/"
    train_path = "train/"
    test_path = "test/"
    train_label_file = "y_train.txt"
    test_label_file = "y_test.txt"

    input_data_path_train = data_path + train_path + signal_path
    input_data_path_test = data_path + test_path + signal_path

    input_data_train_files = list_files(input_data_path_train)
    input_data_test_files = list_files(input_data_path_test)

    if not __debug__:
        print(input_data_train_files)
        print(input_data_test_files)

    input_train = read_data(input_data_train_files)
    input_test = read_data(input_data_test_files)

    train_label_file = data_path + train_path + train_label_file
    label_test_file = data_path + test_path + test_label_file
    label_train = read_label(train_label_file)
    label_test = read_label(label_test_file)

    return (input_train, label_train), (input_test, label_test)

if __name__ == "__main__":
    maybe_download_data()

    print("Begin training model...")
    init_time = time.time()
    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    (x_train, y_train), (x_test, y_test) = get_data()

    # -----------------------------------
    # step2: define parameters for model
    # -----------------------------------
    config = Config(x_train)

    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.time_steps, config.input_dim], name="input")
    Y = tf.placeholder(tf.float32, [None, config.num_classes], name="label")

    label_prob = lstm_net(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables())
    # softmax loss and L2
    cost = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=label_prob, labels=Y, name="cross_entropy"), name="reduce_mean"), l2, name="cost")

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    is_correct = tf.equal(tf.argmax(label_prob, 1), tf.argmax(Y, 1), name="is_correct")
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32), name="accuracy")

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
            sess.run(optimizer, feed_dict={X: x_train[start:end], Y: y_train[start:end]})
        train_time = time.time()
        # Test completely at every epoch: calculate accuracy
        predict_out, accuracy_out, loss_out = sess.run([label_prob, accuracy, cost],
                                                       feed_dict={X: x_test, Y: y_test})
        end_time = time.time()
        if accuracy_out > best_accuracy:
            best_accuracy = accuracy_out
            best_iter = epoch
            save_start_time = time.time()
            saver.save(sess, model_file_name)
            print("Model saved at: {}, takes {:6.4f}s".format(model_file_name, (time.time() - save_start_time)))
        print("Iter:{:3d},".format(epoch)
              + "test_acc: {:6.4f}%,".format(accuracy_out * 100)
              + " loss:{:5.3f},".format(loss_out)
              + " t_train:{:6.4f}s,".format(train_time - begin_time)
              + " t_test:{:6.4f}s,".format(end_time - train_time)
              + " best_test_acc:{:6.4f}".format(best_accuracy * 100) + " (at iter{:3d})".format(best_iter))

    print("best epoch's test accuracy: {:6.4f}".format(best_accuracy * 100) + " at iter:{:3d}".format(best_iter))
    print("finished, takes {:6.4f}s in total".format(time.time() - init_time))
