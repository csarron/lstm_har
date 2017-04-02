#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
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
        print('loaded input tensor shape: ', signal_tensor.shape)
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


def get_tensor_by_op_name(graph, names):
    return [graph.get_operation_by_name(name).outputs[0] for name in names]


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

    model_file_name = "data/lstm_har.model"
    # np.random.seed(0)
    sample_index = np.random.randint(len(y_test))
    print("predicting case:{}".format(sample_index))
    input_test_sample = input_test[np.array([sample_index])]
    y_test_sample = y_test[np.array([sample_index])]
    # print("x{}y{}".format(input_test_sample.shape,y_test_sample.shape))
    init_end_time = time.time()
    print("loading data takes {:6.4f} ms".format((init_end_time - init_time) * 1000))

    if glob.glob(model_file_name + "*"):
        print("begin loading model")
        begin_time = time.time()
        saver = tf.train.import_meta_graph(model_file_name + ".meta")
        session = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
        saver.restore(session, model_file_name)
        end_time = time.time()
        print("model loaded, taking {:6.4f} s".format((end_time - begin_time)))
        begin_time = time.time()
        graph = tf.get_default_graph()
        X, Y = get_tensor_by_op_name(graph, ["input", "label"])
        output, accuracy, cost = get_tensor_by_op_name(graph, ["output", "accuracy", "cost"])

        label_prob = session.run(output, feed_dict={X: input_test_sample, Y: y_test_sample})
        end_time = time.time()

        accuracy_out, loss_out = session.run([accuracy, cost],
                                             feed_dict={X: input_test_sample, Y: y_test_sample})
        print("test_acc: {:5.3f}%,".format(accuracy_out * 100)
              + " time: {:6.4f} ms,".format((end_time - begin_time) * 1000)
              + " cost: {:6.4f}".format(loss_out))
        print("For case:{}, predicted label is: {}".format(sample_index + 1, np.argmax(label_prob) + 1))
        print("Finished, takes {:6.4f} s".format(time.time() - init_time))
    else:
        print("model not exist!")
        exit(1)
