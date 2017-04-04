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

    sample_size = 4
    sample_shape = (2, 2)

    # np.random.seed(0)
    # sample_index = np.random.randint(len(y_test), size=sample_size)
    sample_index = np.arange(sample_size)

    x_test_sample = x_test[sample_index]
    y_test_sample = y_test[sample_index]
    # print("x_sample:{}, y_sample:{}".format(x_test_sample.shape, y_test_sample.shape))
    init_end_time = time.time()
    print("loading data takes {:6.4f} ms".format((init_end_time - init_time) * 1000))
    print("predicting {} cases:".format(sample_size))

    graph = util.load_graph(frozen_model)
    if not __debug__:
        print('\n'.join(map(str, [op.name for op in graph.get_operations()])))
    session = tf.Session(graph=graph)
    begin_time = time.time()
    X, Y = util.get_tensor_by_op_name(graph, ["input", "label"])
    output, accuracy, cost = util.get_tensor_by_op_name(graph, ["output", "accuracy", "cost"])

    label_prob = session.run(output, feed_dict={X: x_test_sample, Y: y_test_sample})
    end_time = time.time()

    np.savetxt("data/label.log", label_prob, '%.4f')

    accuracy_out, loss_out = session.run([accuracy, cost],
                                         feed_dict={X: x_test_sample, Y: y_test_sample})
    print("test_acc: {:5.3f}%,".format(accuracy_out * 100)
          + " time: {:6.4f} ms,".format((end_time - begin_time) * 1000)
          + " cost: {:6.4f}".format(loss_out))

    print("For cases: \n{}".format((sample_index + 1).reshape(sample_shape)))

    print("Predicted labels are: \n{}".format((np.argmax(label_prob, 1) + 1).reshape(sample_shape)))
    print("Finished, takes {:6.4f} s".format(time.time() - init_time))
