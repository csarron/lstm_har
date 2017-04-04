#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import argparse


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
    return [os.path.join(path, data_file) for data_file in os.listdir(path)
            if os.path.isfile(os.path.join(path, data_file))]


def get_tensor_by_op_name(g, names):
    return [g.get_operation_by_name(n).outputs[0] for n in names]


def get_tensor_by_name(g, names):
    return [g.get_tensor_by_name(n).outputs[0] for n in names]


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph


###
"""
w_in 9,32
b_in 32

rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights 64,128
rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases 128
rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights 64,128
rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases 128

w_out 32,6
b_out 6
"""


###


def print_weights(sess):
    g = sess.graph
    weights = ["w_in", "b_in", "w_out", "b_out",
               "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights",
               "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases",
               "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights",
               "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases"]
    for name in weights:
        print("{}:{}".format(name, sess.run(get_tensor_by_op_name(g, name))))


if __name__ == "__main__":
    default_frozen_model = "data/lstm_model.pb"
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model", default=default_frozen_model,
                        type=str, help="Frozen model file to import, default is data/lstm_model.pb")
    parser.add_argument("--model_folder",
                        type=str, help="Model folder to import, default is data")
    args = parser.parse_args()
    model_folder = args.model_folder
    frozen_model = args.frozen_model
    use_frozen_model = True
    if model_folder:
        if os.path.isdir(model_folder):
            use_frozen_model = False
            print("using model folder")
        else:
            print("model folder: {} not exist!".format(frozen_model))
            exit(1)
    elif os.path.isfile(frozen_model):
        print("using frozen model: {}".format(frozen_model))
    else:
        print("model: {} not exist!".format(frozen_model))
        exit(1)

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

    x_test = read_data(input_data_test_files)

    label_test_file = DATA_PATH + TEST + LABEL_TEST_FILE
    y_test = read_label(label_test_file)

    np.random.seed(0)
    sample_size = 100
    sample_index = np.random.randint(len(y_test), size=sample_size)
    x_test_sample = x_test[sample_index]
    y_test_sample = y_test[sample_index]
    # print("x_sample:{}, y_sample:{}".format(x_test_sample.shape, y_test_sample.shape))
    init_end_time = time.time()
    print("loading data takes {:6.4f} ms".format((init_end_time - init_time) * 1000))
    print("predicting {} cases:".format(sample_size))

    if not use_frozen_model:
        print("begin loading model")
        begin_time = time.time()
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        saver = tf.train.import_meta_graph(input_checkpoint + ".meta")
        session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        saver.restore(session, input_checkpoint)
        end_time = time.time()
        print("model loaded, taking {:6.4f} s".format((end_time - begin_time)))
        graph = tf.get_default_graph()
        # begin_time = time.time()
        # X, Y = get_tensor_by_op_name(graph, ["input", "label"])
        # output, accuracy, cost = get_tensor_by_op_name(graph, ["output", "accuracy", "cost"])

    else:
        graph = load_graph(frozen_model)
        if not __debug__:
            print('\n'.join(map(str, [op.name for op in graph.get_operations()])))
        session = tf.Session(graph=graph)

    begin_time = time.time()
    X, Y = get_tensor_by_op_name(graph, ["input", "label"])
    output, accuracy, cost = get_tensor_by_op_name(graph, ["output", "accuracy", "cost"])

    label_prob = session.run(output, feed_dict={X: x_test_sample, Y: y_test_sample})
    end_time = time.time()

    accuracy_out, loss_out = session.run([accuracy, cost],
                                         feed_dict={X: x_test_sample, Y: y_test_sample})
    print("test_acc: {:5.3f}%,".format(accuracy_out * 100)
          + " time: {:6.4f} ms,".format((end_time - begin_time) * 1000)
          + " cost: {:6.4f}".format(loss_out))

    print("For cases: \n{}".format((sample_index + 1).reshape((10, 10))))

    print("Predicted label are: \n{}".format((np.argmax(label_prob, 1) + 1).reshape((10, 10))))

    # export network weights and biases to text files
    weights = ["w_in", "b_in", "w_out", "b_out",
               "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights",
               "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases",
               "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights",
               "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases"]
    for name in weights:
        v = session.run("{}:0".format(name))
        var_file_name = "data/{}.csv".format(name.replace("/", "_"))
        print("save {} to file: {}".format(name, var_file_name))
        np.savetxt(var_file_name, v, delimiter=", ")

    print("Finished, takes {:6.4f} s".format(time.time() - init_time))
