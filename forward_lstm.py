from __future__ import print_function
import numpy as np
import util

x_test, y_test = util.get_data("test")
time_steps = len(x_test[0])
input_dim = len(x_test[0][0])

sample_size = 1
sample_shape = (1, 1)

sample_index = np.arange(sample_size)

x = x_test[sample_index]
x = np.reshape(x, [-1, input_dim])
y = y_test[sample_index]

print("x: {}, y: {}".format(x.shape, y.shape))

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
    weights[name] = np.loadtxt(var_file_name, delimiter=", ")
    # print("{}: {}".format(name, weights[name]))

inputs = np.maximum(np.dot(x, weights["w_in"]) + weights["b_in"], 0)
np.savetxt("inputs.log", inputs, '%.4f')

for step in xrange(time_steps):
    print(step)
