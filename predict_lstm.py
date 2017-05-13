#!/usr/bin/python
from __future__ import print_function
import argparse
import tensorflow as tf
import numpy as np
import os
import time
import data_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=2,
                        help="lay size of the LSTM model")
    parser.add_argument("--unit", type=int, default=32,
                        help="hidden unit of the LSTM model")
    args = parser.parse_args()
    frozen_model = "data/{}layer{}unit.pb".format(args.layer, args.unit)

    if os.path.isfile(frozen_model):
        print("using frozen model: {}".format(frozen_model))
    else:
        print("model: {} not exist!".format(frozen_model))
        exit(1)

    init_time = time.time()
    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    x_test, y_test = data_util.get_data("test")

    # np.random.seed(0)
    # sample_index = np.random.randint(len(y_test), size=sample_size)
    sample_index = np.arange(len(y_test))

    x_test_sample = x_test[sample_index]
    y_test_sample = y_test[sample_index]
    # print("x_sample:{}, y_sample:{}".format(x_test_sample.shape, y_test_sample.shape))
    init_end_time = time.time()
    print("loading data takes {:6.4f} ms".format((init_end_time - init_time) * 1000))
    print("predicting cases:")

    with tf.gfile.GFile(frozen_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                            name="", op_dict=None, producer_op_list=None)
    if not __debug__:
        print('\n'.join(map(str, [op.name for op in graph.get_operations()])))
    session = tf.Session(graph=graph)
    begin_time = time.time()
    X = graph.get_operation_by_name("input").outputs[0]
    output = graph.get_operation_by_name("output").outputs[0]

    label_prob = session.run(output, feed_dict={X: x_test_sample})
    end_time = time.time()

    print("For cases: \n{}".format((sample_index + 1)))
    labels_predicted = (np.argmax(label_prob, 1) + 1)
    labels = (np.argmax(y_test_sample, 1) + 1)

    print("test_acc: {:5.3f}%,".format(np.sum(labels == labels_predicted) * 1.0 * 100 / len(y_test))
          + " time: {:6.4f} ms,".format((end_time - begin_time) * 1000))

    print("Predicted labels are: \n{}".format(labels_predicted))
    print("Finished, takes {:6.4f} s".format(time.time() - init_time))

    if not __debug__:
        np.savetxt("data/label_prob.log", label_prob, '%.7e')
        np.savetxt("data/x.log", np.reshape(x_test_sample, [-1, 9]), '%.7e')

        inputs = session.run("input:0", feed_dict={X: x_test_sample})
        np.savetxt("data/input.log", np.reshape(inputs, [-1, 9]), '%.7e')

        inputs = session.run("transpose:0", feed_dict={X: x_test_sample})
        np.savetxt("data/transpose.log", np.reshape(inputs, [-1, 9]), '%.7e')

        inputs = session.run("reshape:0", feed_dict={X: x_test_sample})
        np.savetxt("data/reshape.log", inputs, '%.7e')

        inputs = session.run("relu:0", feed_dict={X: x_test_sample})
        np.savetxt("data/relu.log", inputs, '%.7e')
        np.savetxt("data/labels.log", labels_predicted, fmt="%d")
