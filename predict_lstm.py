#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import util

if __name__ == "__main__":
    frozen_model = "data/lstm_model.pb"
    if os.path.isfile(frozen_model):
        print("using frozen model: {}".format(frozen_model))
    else:
        print("model: {} not exist!".format(frozen_model))
        exit(1)

    init_time = time.time()
    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    x_test, y_test = util.get_data("test")

    # np.random.seed(0)
    # sample_index = np.random.randint(len(y_test), size=sample_size)
    sample_index = np.arange(len(y_test))

    x_test_sample = x_test[sample_index]
    y_test_sample = y_test[sample_index]
    # print("x_sample:{}, y_sample:{}".format(x_test_sample.shape, y_test_sample.shape))
    init_end_time = time.time()
    print("loading data takes {:6.4f} ms".format((init_end_time - init_time) * 1000))
    print("predicting cases:")

    graph = util.load_graph(frozen_model)
    if not __debug__:
        print('\n'.join(map(str, [op.name for op in graph.get_operations()])))
    session = tf.Session(graph=graph)
    begin_time = time.time()
    X, Y = util.get_tensor_by_op_name(graph, ["input", "label"])
    output, accuracy, cost = util.get_tensor_by_op_name(graph, ["output", "accuracy", "cost"])

    label_prob = session.run(output, feed_dict={X: x_test_sample, Y: y_test_sample})
    end_time = time.time()

    np.savetxt("data/label_prob.log", label_prob, '%.7e')

    inputs = session.run("input:0", feed_dict={X: x_test_sample, Y: y_test_sample})
    np.savetxt("data/input.log", np.reshape(inputs, [-1, 9]), '%.7e')

    inputs = session.run("transpose:0", feed_dict={X: x_test_sample, Y: y_test_sample})
    np.savetxt("data/transpose.log", np.reshape(inputs, [-1, 9]), '%.7e')

    inputs = session.run("reshape:0", feed_dict={X: x_test_sample, Y: y_test_sample})
    np.savetxt("data/reshape.log", inputs, '%.7e')

    inputs = session.run("relu:0", feed_dict={X: x_test_sample, Y: y_test_sample})
    np.savetxt("data/relu.log", inputs, '%.7e')

    accuracy_out, loss_out = session.run([accuracy, cost],
                                         feed_dict={X: x_test_sample, Y: y_test_sample})
    print("test_acc: {:5.3f}%,".format(accuracy_out * 100)
          + " time: {:6.4f} ms,".format((end_time - begin_time) * 1000)
          + " cost: {:6.4f}".format(loss_out))

    print("For cases: \n{}".format((sample_index + 1)))
    labels_predicted = (np.argmax(label_prob, 1) + 1)
    np.savetxt("data/labels.log", labels_predicted, fmt="%d")

    print("Predicted labels are: \n{}".format((np.argmax(label_prob, 1) + 1)))
    print("Finished, takes {:6.4f} s".format(time.time() - init_time))
