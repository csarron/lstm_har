from __future__ import print_function
import numpy as np
import time
import util

weight_names = ["w_in", "b_in", "w_out", "b_out",
                "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights",
                "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases",
                "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights",
                "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases"]

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

weights = {}
for name in weight_names:
    var_file_name = "data/{}.csv".format(name.replace("/", "_"))
    weights[name] = np.loadtxt(var_file_name, delimiter=",")
    # print("{}: {}".format(name, weights[name]))


def sigmoid(x_): return 1 / (1 + np.exp(-x_))


def calc_cell_one_step(in_, c_, h_, l):
    # print("h:\n{}".format(h))
    # print("x_step:\n{}".format(x_step))
    concat = np.concatenate([in_, h_], 1) \
                 .dot(weights["rnn/multi_rnn_cell/cell_{}/basic_lstm_cell/weights".format(l)]) \
             + weights["rnn/multi_rnn_cell/cell_{}/basic_lstm_cell/biases".format(l)]
    # print("concat:{}".format(concat.shape))
    i, j, f, o = np.split(concat, 4, axis=1)
    # print("i:{}, j:{}, f:{}, o:{}".format(i.shape, j.shape, f.shape, o.shape))
    new_c = (c_ * sigmoid(f + 1) + sigmoid(i) * np.tanh(j))
    new_h = np.tanh(new_c) * sigmoid(o)
    return new_c, new_h


def predict(x_):
    inputs = np.maximum(np.dot(x_, weights["w_in"]) + weights["b_in"], 0)
    # np.savetxt("data/inputs_np.log", inputs, '%.8f')
    hidden_units = len(weights["b_in"])

    inputs = np.split(inputs, time_steps, 0)
    outputs = []
    for layer in xrange(layer_size):
        c = np.zeros((1, hidden_units))
        h = np.zeros((1, hidden_units))
        for step in xrange(time_steps):
            input_ = inputs[step]
            c, h = calc_cell_one_step(input_, c, h, layer)
            inputs[step] = h
            outputs.append(h)
    out_prob = np.dot(outputs[-1], weights["w_out"]) + weights["b_out"]
    # print("out_prob:{}\n{}".format(out_prob.shape, out_prob))
    # np.savetxt("data/label_prob_np.log", out_prob, '%.4f')

    return np.argmax(out_prob) + 1

    # np.savetxt("data/labels_np.log", labels_predicted, fmt="%d")


if __name__ == "__main__":
    start_time = time.time()

    x_test, y_test = util.get_data("test")
    time_steps = len(x_test[0])
    input_dim = len(x_test[0][0])

    layer_size = 2

    sample_index = np.random.randint(0, len(y_test))

    labels_predicted = [predict(x_test[i]) for i in xrange(len(y_test))]
    labels = np.argmax(y_test, 1) + 1
    print("label:\n{}\nY:\n{}".format(np.asarray(labels_predicted), labels))
    print("accuracy: {}".format(np.sum(labels == np.asarray(labels_predicted)) * 1.0 * 100 / len(y_test)))
    np.savetxt("data/labels_np.log", labels_predicted, fmt="%d")

    print("Finished, takes {:6.4f} s".format(time.time() - start_time))
